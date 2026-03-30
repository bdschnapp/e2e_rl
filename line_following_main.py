from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

import numpy as np
import os

from Models.CNNFeatureExtractor import CNNFeatureExtractor
from Models.AutoEncoder import train_autoencoder


class RenderCallback(BaseCallback):
    """
    A custom callback that renders the environment every N steps.

    :param render_freq: The frequency to render the environment (e.g., 1 for every step)
    """

    def __init__(self, render_freq: int, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        # Render the environment every `render_freq` steps
        if self.n_calls % self.render_freq == 0:
            # self.training_env.render() is vectorized, we need to call the underlying env
            self.training_env.get_attr('render')[0]()
        return True  # Must return True to continue training


class ObstacleCallback(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles_low = 0
        self.obstacles_high = 0
        self.freq = 2000

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.obstacles_low =min(self.obstacles_low + 1, 5)
            self.training_env.set_attr('obstacles_low', self.obstacles_low)

            self.obstacles_high = min(self.obstacles_high + 2, 10)
            self.training_env.set_attr('obstacles_high', self.obstacles_high)
        return True


class NormalizedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_normalized_score = -np.inf
        self.k = 1

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Only run logic right after an evaluation
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if len(self.evaluations_results) == 0:
                return result

            rewards = self.evaluations_results[-1]
            lengths = self.evaluations_length[-1]

            mean_reward = float(np.mean(rewards))
            mean_length = float(np.mean(lengths))

            if mean_length <= 0:
                return result

            normalized_score = mean_reward / mean_length

            if normalized_score > self.best_normalized_score:
                self.best_normalized_score = normalized_score

                save_path = os.path.join(
                    self.best_model_save_path,
                    f"norm{self.k}"
                )
                self.model.save(save_path)
                self.k += 1

                print(
                    f"[Eval] New best normalized model | "
                    f"reward={mean_reward:.2f}, "
                    f"len={mean_length:.2f}, "
                    f"score={normalized_score:.3f}"
                )

        return result


def main(render_mode="human", save_path="./models/", timesteps=200_000):
    env = Monitor(LineFollowingEnv(render_mode=render_mode))

    # Create the action noise object for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))

    # policy_kwargs = dict(
    #     features_extractor_class=CNNFeatureExtractor,
    #     features_extractor_kwargs=dict()
    # )
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=0,
        device='cuda',
        # policy_kwargs=policy_kwargs,
        buffer_size=200_000
    )

    cbs = []
    if render_mode:
        cbs.append(RenderCallback(render_freq=1))

    eval_env = LineFollowingEnv(render_mode='human')
    if hasattr(eval_env, 'obstacles_low'):
        eval_env.obstacles_low = 5
        eval_env.obstacles_high = 10
        cbs.append(ObstacleCallback())
    eval_env = Monitor(eval_env)

    cbs.append(
        EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path="./logs/",
            eval_freq=1_000,
            deterministic=True,
            render=False
        )
    )
    cbs.append(
        NormalizedEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_path, "normalized"),
            log_path="./eval_logs",
            eval_freq=3_000,
            n_eval_episodes=10,
            deterministic=True,
        )
    )

    # Train the model
    model.learn(total_timesteps=timesteps, callback=cbs)


if __name__ == "__main__":
    # # train forward model
    # from Environments.LineFollowing import StateObservationLineFollowingEnv as LineFollowingEnv
    # main(render_mode=None,
    #      save_path="./models/LineFollowing/Forward/",
    #      timesteps=30_000)
    #
    # # train reverse model
    # from Environments.LineFollowing import ReverseStateObservationLineFollowingEnv as LineFollowingEnv
    # main(render_mode=None,
    #      save_path="./models/LineFollowing/Reverse/",
    #      timesteps=60_000)

    # train forward model with obstacles
    from Environments.ObstacleAvoidance import ObstacleAvoidanceEnv as LineFollowingEnv
    main(render_mode=None,
         save_path="./models/ObstacleAvoidance/Forward/",
         timesteps=180_000)

    # # train reverse model with obstacles
    # from Environments.ObstacleAvoidance import ReverseObstacleAvoidanceEnv as LineFollowingEnv
    # main(render_mode=None,
    #      save_path="./models/ObstacleAvoidance/Reverse/",
    #      timesteps=180_000)
