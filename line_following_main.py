from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

import numpy as np

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


def main(render_mode="human", save_path="./models/LineFollowing/"):
    env = LineFollowingEnv(render_mode=render_mode)

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
        verbose=1,
        device='cuda',
        # policy_kwargs=policy_kwargs,
        buffer_size=200_000
    )
    cbs = []
    if render_mode:
        cbs.append(RenderCallback(render_freq=1))
    eval_env = LineFollowingEnv(render_mode='human')
    cbs.append(EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=1_000,
        deterministic=True,
        render=False
    ))

    # Train the model
    model.learn(total_timesteps=200_000, callback=cbs)


if __name__ == "__main__":
    # train forward model
    # from Environments.LineFollowing import StateObservationLineFollowingEnv as LineFollowingEnv
    # main(render_mode=None, save_path="./models/LineFollowing/Forward/")

    # train reverse model
    from Environments.LineFollowing import ReverseStateObservationLineFollowingEnv as LineFollowingEnv
    main(render_mode=None, save_path="./models/LineFollowing/Reverse/")
