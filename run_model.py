import argparse

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from Environments.LineFollowing import forward_pure_pursuit, reverse_pure_pursuit


def fpp(n_episodes: int, render: bool):
    env = LineFollowingEnv(render_mode="human" if render else None)
    print("Forward Pure Pursuit Evaluation:")
    for ep in range(1, n_episodes + 1):
        total_reward, steps = forward_pure_pursuit(env, render=render)
        print(
            f"Episode {ep:03d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Length: {steps:4d}"
        )


def rpp(n_episodes: int, render: bool):
    env = LineFollowingEnv(render_mode="human" if render else None)
    print("Reverse Pure Pursuit Evaluation:")
    for ep in range(1, n_episodes + 1):
        total_reward, steps = reverse_pure_pursuit(env, render=render)
        print(
            f"Episode {ep:03d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Length: {steps:4d}"
        )


def run(model_path: str, n_episodes: int, render: bool):
    env = LineFollowingEnv(render_mode="human" if render else None)
    env = Monitor(env)

    model = TD3.load(model_path, env=env)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if render:
                env.render()

        print(
            f"Episode {ep:03d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Length: {steps:4d}"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reverse", action="store_true")

    args = parser.parse_args()

    if args.reverse:
        print("Using Reverse Driving Model")
        from Environments.LineFollowing import ReverseStateObservationLineFollowingEnv as LineFollowingEnv
    else:
        print("Using Forward Driving Model")
        from Environments.LineFollowing import StateObservationLineFollowingEnv as LineFollowingEnv

    if args.model == 'fpp':
        fpp(args.episodes, args.render)

    elif args.model == 'rpp':
        rpp(args.episodes, args.render)

    else:
        run(args.model, args.episodes, args.render)