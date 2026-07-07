"""Train an SB3 algorithm on the batched GPU env (via the VecEnv adapter).

Vetted TD3 / SAC / PPO / DQN on the fast batched env — the algorithm-ablation
trainer. Reports wall-clock + eval completion so it can be compared head-to-head
with the hand-rolled `batched/td3.py`.

    TTRL_BACKEND=cupy python scripts/train_sb3.py --algo td3 \
        --cell forward:lidar24:multiplicative --preset truck --fixed_speed --mild_paths \
        --n_envs 256 --steps 500000
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.batched.backend import to_numpy  # noqa: E402

# reuse the cell/cfg builder from the custom runner
sys.path.insert(0, os.path.dirname(__file__))
from run_ablations_gpu import cell_to_cfg  # noqa: E402


JACKKNIFE_RAD = 1.4  # ~80 deg; the env terminates hard at pi/2


def evaluate(model, env, steps=1500):
    """Full thesis metric set from a deterministic rollout. Trailer CTE (|e_y_t|) and
    hitch (|gamma|) are read directly from the obs vector (trailer layout:
    [s, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2, ...])."""
    o = env.reset()
    N = env.num_envs
    ep_ret = np.zeros(N); ep_cte = np.zeros(N); ep_len = np.zeros(N); ep_maxh = np.zeros(N)
    rets, comps, ctes, hitches, jacks = [], [], [], [], []
    _vec = lambda ob: ob["vector"] if isinstance(ob, dict) else ob  # BEV: metrics live in the vector part
    for _ in range(steps):
        v = _vec(o)
        hitch = np.abs(v[:, 1]); cte = np.abs(v[:, 4])   # pre-step (current) state
        ep_maxh = np.maximum(ep_maxh, hitch); ep_cte += cte; ep_len += 1
        a, _ = model.predict(o, deterministic=True)
        o, r, dones, infos = env.step(a)
        ep_ret += r
        for i in np.nonzero(dones)[0]:
            rets.append(ep_ret[i]); comps.append(float(infos[i].get("is_success", 0.0)))
            ctes.append(ep_cte[i] / max(ep_len[i], 1)); hitches.append(ep_maxh[i])
            jacks.append(1.0 if ep_maxh[i] > JACKKNIFE_RAD else 0.0)
            ep_ret[i] = ep_cte[i] = ep_len[i] = ep_maxh[i] = 0.0
    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(completion=m(comps), trailer_cte=m(ctes), max_hitch=m(hitches),
                jackknife=m(jacks), ep_return=m(rets), n_episodes=len(rets))


class EvalCurveCallback:
    """Periodic eval → records the full metric set vs. timesteps (the learning curve)
    AND tracks the BEST checkpoint (by completion desc, then trailer-CTE asc — both
    reward-agnostic) so the ablation compares each run's *best* policy, not its final
    (crucial for reverse, where a run can peak then collapse into a jackknife).
    Optionally saves the best model. Wraps SB3 BaseCallback."""

    def __new__(cls, eval_env, eval_freq, eval_steps=400, threshold=0.9, best_model_path=None):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.eval_env = eval_env; self.eval_freq = eval_freq
                self.eval_steps = eval_steps; self.threshold = threshold
                self.best_model_path = best_model_path
                self.curve = []; self.steps_to_threshold = None; self._last = 0
                self.best = None; self._best_key = None

            def _on_step(self):
                if self.num_timesteps - self._last >= self.eval_freq:
                    self._last = self.num_timesteps
                    mt = evaluate(self.model, self.eval_env, self.eval_steps)
                    mt["timesteps"] = int(self.num_timesteps)
                    self.curve.append(mt)
                    if self.steps_to_threshold is None and mt["completion"] >= self.threshold:
                        self.steps_to_threshold = int(self.num_timesteps)
                    cte = mt["trailer_cte"] if mt["trailer_cte"] == mt["trailer_cte"] else 1e9
                    key = (mt["completion"], -cte)  # best-by completion then low CTE
                    if self._best_key is None or key > self._best_key:
                        self._best_key = key; self.best = dict(mt)
                        if self.best_model_path is not None:
                            try:
                                self.model.save(self.best_model_path)
                            except Exception:
                                pass
                return True
        return _CB()


DISCRETE_ALGOS = {"dqn"}  # ppo can be discrete too via --discrete


def _td3_action_noise(env):
    """Per-dimension Gaussian exploration noise from the env's action config, so
    TD3 actually explores (SB3's default is None) with LOW noise on the stop channel
    (stop_noise_sigma) — an unintended-stop guard the config previously couldn't set."""
    from stable_baselines3.common.noise import NormalActionNoise
    shape = getattr(env.action_space, "shape", None)
    if not shape:                       # Discrete (DQN) has no action noise
        return None
    n = shape[0]
    cfg = getattr(getattr(env, "env", None), "cfg", None)
    steer_sig = cfg.action.steer_noise_sigma if cfg else 0.05
    stop_sig = cfg.action.stop_noise_sigma if cfg else 0.05
    sig = np.full(n, steer_sig, dtype=np.float32)
    if n > 1:
        sig[1:] = stop_sig
    return NormalActionNoise(mean=np.zeros(n, dtype=np.float32), sigma=sig)


def _bev_extractor_class():
    """SB3 features extractor for the BEV Dict obs {vector, image}: a small CNN over
    the 32x32 image (NatureCNN's 8x8/s4 stack collapses below ~36px, so we use a
    3x3/stride-2 tower) concatenated with the low-dim state vector. Returned lazily
    so torch is only imported when a BEV cell is actually trained."""
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class BevCombinedExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, cnn_features=128):
            img_space = observation_space.spaces["image"]
            vec_dim = int(observation_space.spaces["vector"].shape[0])
            super().__init__(observation_space, features_dim=cnn_features + vec_dim)
            c = int(img_space.shape[0])
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, 3, stride=2, padding=1), nn.ReLU(),   # 32 -> 16
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8 -> 4
                nn.Flatten(),
            )
            with th.no_grad():
                n_flat = self.cnn(th.zeros(1, *img_space.shape)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flat, cnn_features), nn.ReLU())

        def forward(self, obs):
            feat = self.linear(self.cnn(obs["image"]))
            return th.cat([feat, obs["vector"]], dim=1)

    return BevCombinedExtractor


def _obs_policy(env):
    """(policy_name, extra_policy_kwargs) — MultiInputPolicy + BEV CNN extractor for
    a Dict obs space, else plain MlpPolicy. Keeps the state/lidar path unchanged."""
    import gymnasium as gym
    if isinstance(env.observation_space, gym.spaces.Dict):
        return "MultiInputPolicy", {"features_extractor_class": _bev_extractor_class()}
    return "MlpPolicy", {}


def build_model(algo, env, lr, batch_size, seed):
    """Stable, validated hyperparameters (frozen 'nuisance' config).

    Off-policy (TD3/SAC/DQN) use SB3 DEFAULTS + a modest lr + action noise — the
    regime proven in the sibling tractor_trailer_rl repo. The earlier throughput-tuned
    overrides (lr 2e-3, batch 4096, 32 grad-steps/step) were BRITTLE: they diverged
    once the reward magnitude grew or the stop action was added. SB3 defaults (batch
    256, buffer 1M, train_freq=(1,episode), gradient_steps=-1, net [400,300]) are the
    stable, reproducible choice for the ablation; the env's throughput contribution
    stands separately on the benchmark + custom GPU-TD3 demo."""
    from stable_baselines3 import TD3, SAC, PPO, DQN
    policy, pk_obs = _obs_policy(env)   # MultiInputPolicy + BEV CNN when obs is a Dict
    common = dict(policy=policy, env=env, seed=seed, device="cuda", verbose=0)
    if algo == "td3":
        return TD3(learning_rate=lr, learning_starts=10_000,
                   action_noise=_td3_action_noise(env),
                   policy_kwargs=pk_obs or None, **common)
    if algo == "sac":
        return SAC(learning_rate=lr, learning_starts=10_000,
                   policy_kwargs=pk_obs or None, **common)
    if algo == "ppo":  # on-policy: many parallel envs are good; keep the working config
        return PPO(learning_rate=lr, n_steps=32, batch_size=batch_size, n_epochs=5,
                   gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                   policy_kwargs=dict(net_arch=[256, 256], **pk_obs), **common)
    if algo == "dqn":  # off-policy discrete: value-based, no continuous-actor cliff, so
        # the larger batch it benefits from is safe (SB3 default batch=32 is too small
        # here — DQN peaked then declined). Restore the bigger-batch regime that worked.
        return DQN(learning_rate=lr, batch_size=batch_size, buffer_size=300_000,
                   learning_starts=10_000, train_freq=(1, "step"), gradient_steps=16,
                   target_update_interval=2_000, exploration_fraction=0.2,
                   exploration_final_eps=0.05, gamma=0.99,
                   policy_kwargs=dict(net_arch=[256, 256], **pk_obs), **common)
    raise ValueError(algo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["td3", "sac", "ppo", "dqn"], default="td3")
    ap.add_argument("--discrete", action="store_true",
                    help="use the Discrete(n_steer_bins*2) action space (auto for dqn)")
    ap.add_argument("--cell", default="forward:lidar24:multiplicative")
    ap.add_argument("--preset", choices=["lab", "truck", "shunt"], default="shunt")
    ap.add_argument("--fixed_speed", action="store_true")
    ap.add_argument("--mild_paths", action="store_true")
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate; if unset, per-algo default (PPO 1e-3, off-policy 2e-3). "
                         "PPO collapses at 2e-3 and is too slow at 3e-4 — it needs its own LR.")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV for the eval learning curve")
    ap.add_argument("--eval_every", type=int, default=None,
                    help="eval frequency in timesteps (default: steps//8)")
    args = ap.parse_args()

    cfg = cell_to_cfg(args.cell, preset=args.preset, fixed_speed=args.fixed_speed,
                      mild_paths=args.mild_paths)
    # 2-D stop-capable action for every algo (symmetric 2x2 + carries to the
    # obstacle env): discrete => Discrete(n*2)=steer x {go,stop}; continuous =>
    # STOP_SIGNAL [steer, stop]. Dormant in pure lane-following.
    from dataclasses import replace
    from tractor_trailer_rl.config import ActionMode
    if args.discrete or args.algo in DISCRETE_ALGOS:
        cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.DISCRETE))
    else:
        cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    lr = args.lr if args.lr is not None else 1e-3  # proven-stable for all algos
    from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
    env = SB3BatchedVecEnv(cfg, args.n_envs, path_pool_size=1024)
    model = build_model(args.algo, env, lr, args.batch_size, args.seed)
    eval_env = SB3BatchedVecEnv(cfg, min(args.n_envs, 256), path_pool_size=512)
    eval_freq = args.eval_every or max(args.steps // 8, 1)
    cb = EvalCurveCallback(eval_env, eval_freq)
    print(f"# SB3 {args.algo.upper()} backend={backend.get_backend()} n_envs={args.n_envs} "
          f"steps={args.steps} batch={args.batch_size} lr={lr} preset={args.preset} cell={args.cell}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=cb)
    train_s = time.perf_counter() - t0

    fin = evaluate(model, eval_env, steps=800)
    best = cb.best or fin
    stt = cb.steps_to_threshold
    print(f"# {args.algo}: train {train_s:.0f}s | BEST completion={best['completion']:.2f} "
          f"trailer_cte={best['trailer_cte']:.3f}m max_hitch={best['max_hitch']:.3f}rad "
          f"jackknife={best['jackknife']:.2f} @{best.get('timesteps')} "
          f"| final_compl={fin['completion']:.2f} | steps_to_90%={stt if stt is not None else 'NA'}")
    if args.out:
        import csv
        keys = ["timesteps", "completion", "trailer_cte", "max_hitch", "jackknife",
                "ep_return", "n_episodes"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for row in cb.curve:
                w.writerow({k: row.get(k) for k in keys})
        print(f"# wrote learning curve -> {args.out} ({len(cb.curve)} evals)")


if __name__ == "__main__":
    main()
