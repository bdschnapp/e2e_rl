"""TD3 training entry (clean replacement for e2e_rl/train.py + the lab trainers).

A single `train(cfg, ...)` builds the vectorised env from a Config, a TD3 agent
with action-mode-appropriate exploration noise, eval callbacks, and trains. No
monkeypatching: behaviour is fully determined by the Config.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def make_env_fn(cfg, seed=None):
    from ..envs.base import LaneFollowingEnv

    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        env = LaneFollowingEnv(cfg)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk


def _completion_eval_callback(cfg, *, out, n_eval_episodes, eval_freq, n_envs, seed, verbose=0):
    """EvalCallback that saves best_model by COMPLETION RATE first, mean reward
    second (lexicographic) instead of raw mean reward. A high-reward checkpoint can
    still be unreliable (e.g. great on straights, collides on curves); the metric
    we actually want at deploy is "finishes the most episodes, then tracks best".

    completion = episode ended by TRUNCATION (survived the full horizon) OR success
    (reached goal) — NOT an early terminated() (jackknife / OOB / collision / fired
    STOP). Writes logs/evaluations.npz with timesteps, results (mean reward),
    ep_lengths, AND completions (the new selection metric)."""
    from stable_baselines3.common.callbacks import BaseCallback
    from ..envs.base import LaneFollowingEnv

    class _CompletionEval(BaseCallback):
        def __init__(self):
            super().__init__(verbose)
            self.eval_freq = max(1, eval_freq // n_envs)
            self.n_eval_episodes = n_eval_episodes
            self.save_path = str(out)
            self.log_dir = str(Path(out) / "logs")
            self.seed = 10_000 + seed
            self.best_completion = -1.0
            self.best_reward = -np.inf
            self._env = None
            self.timesteps, self.results, self.ep_lengths, self.completions = [], [], [], []

        def _eval_once(self):
            if self._env is None:
                self._env = LaneFollowingEnv(cfg)
            rewards, lengths, completed = [], [], []
            for i in range(self.n_eval_episodes):
                obs, _ = self._env.reset(seed=self.seed + i)
                done = term = trunc = False
                R = 0.0; n = 0; info = {}
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = self._env.step(np.asarray(a).flatten())
                    R += float(r); n += 1; done = term or trunc
                # completed = survived to the horizon (truncated) or reached the goal;
                # an early terminated() (crash/jackknife/STOP) is NOT a completion.
                completed.append(bool(trunc and not term) or bool(info.get("success", False)))
                rewards.append(R); lengths.append(n)
            return float(np.mean(completed)), float(np.mean(rewards)), float(np.mean(lengths))

        def _on_step(self):
            if self.n_calls % self.eval_freq != 0:
                return True
            cr, mr, ml = self._eval_once()
            self.timesteps.append(int(self.num_timesteps))
            self.results.append(mr); self.ep_lengths.append(ml); self.completions.append(cr)
            os.makedirs(self.log_dir, exist_ok=True)
            np.savez(os.path.join(self.log_dir, "evaluations.npz"),
                     timesteps=np.array(self.timesteps),
                     results=np.array(self.results)[:, None],
                     ep_lengths=np.array(self.ep_lengths)[:, None],
                     completions=np.array(self.completions)[:, None])
            # Lexicographic: completion rate first, mean reward as the tiebreak.
            improved = (cr > self.best_completion + 1e-9) or (
                abs(cr - self.best_completion) <= 1e-9 and mr > self.best_reward)
            if improved:
                self.best_completion, self.best_reward = cr, mr
                os.makedirs(self.save_path, exist_ok=True)
                self.model.save(os.path.join(self.save_path, "best_model"))
            if self.verbose:
                print(f"[eval] t={self.num_timesteps} completion={cr:.2f} "
                      f"reward={mr:.1f} len={ml:.0f} (best cr={self.best_completion:.2f})",
                      flush=True)
            return True

    return _CompletionEval()


def build_model(cfg, vec_env, device="auto", learning_starts=10_000, tensorboard_log=None,
                learning_rate=1e-3):
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from ..actions import modes as action_modes

    n_actions = vec_env.action_space.shape[-1]
    sigma = action_modes.noise_sigma(cfg, n_actions)
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma)
    return TD3(
        "MlpPolicy", vec_env, action_noise=noise, device=device,
        learning_starts=learning_starts, learning_rate=learning_rate,
        verbose=0, tensorboard_log=tensorboard_log,
    )


def train(cfg, *, timesteps=200_000, n_envs=8, out_dir="runs/model",
          eval_freq=20_000, n_eval_episodes=10, device="auto", seed=0, learning_rate=1e-3):
    """Train a TD3 policy for `cfg`. Writes best_model.zip + final.zip under out_dir."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    VecCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = VecCls([make_env_fn(cfg, seed=seed + i) for i in range(n_envs)])

    model = build_model(cfg, train_env, device=device, tensorboard_log=None,
                        learning_rate=learning_rate)
    eval_cb = _completion_eval_callback(
        cfg, out=out, n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq, n_envs=n_envs, seed=seed)
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    model.save(str(out / "final"))
    train_env.close()
    return model


def finetune(init_zip, cfg, *, timesteps=150_000, n_envs=8, out_dir="runs/finetune",
             eval_freq=20_000, n_eval_episodes=10, device="auto", seed=0, learning_rate=None):
    """Warm-start fine-tune: load a trained .zip and continue training under a new
    Config (e.g. mild baseline -> full path mixture with corners). Obs/action
    spaces must match the checkpoint (same vehicle_kind/direction/obs)."""
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from ..actions import modes as action_modes

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    VecCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = VecCls([make_env_fn(cfg, seed=seed + i) for i in range(n_envs)])

    model = TD3.load(str(init_zip), env=train_env, device=device)
    # Re-arm exploration noise for the new phase (the loaded noise reflects the
    # prior phase). Keep it small on the stop dim.
    n_actions = train_env.action_space.shape[-1]
    model.action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=action_modes.noise_sigma(cfg, n_actions))
    if learning_rate is not None:
        model.learning_rate = learning_rate
        model._setup_lr_schedule()

    eval_cb = _completion_eval_callback(
        cfg, out=out, n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq, n_envs=n_envs, seed=seed)
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False,
                reset_num_timesteps=True)
    model.save(str(out / "final"))
    train_env.close()
    return model
