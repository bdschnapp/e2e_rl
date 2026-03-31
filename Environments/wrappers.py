"""
Gymnasium environment wrappers for training curriculum strategies.

RetryOnFailureWrapper
---------------------
On a failed episode (collision, jackknife, or OOB before reaching the goal),
the wrapper replays the **exact same path and spawn** by re-seeding with the
same seed used at the start of that episode.  On success or timeout the next
reset draws a fresh random scenario as usual.

This implements a simple "failure replay curriculum": the agent is forced to
keep attempting the same hard layout until it succeeds, rather than escaping
by getting a new (possibly easier) random scenario.  It is most beneficial for
open-loop-unstable tasks such as reverse driving, where random resets provide
insufficient exposure to any single difficult configuration.

Usage
-----
    env = RetryOnFailureWrapper(make_env(...))

    # Works transparently with SB3 VecEnvs because the retry logic is inside
    # the wrapper and the VecEnv just sees normal reset() / step() calls.

    from stable_baselines3.common.vec_env import SubprocVecEnv
    def _make():
        return RetryOnFailureWrapper(make_env(...))
    train_env = SubprocVecEnv([_make] * n_envs)
"""

import numpy as np
import gymnasium


class RetryOnFailureWrapper(gymnasium.Wrapper):
    """
    Replay the same scenario when an episode fails.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap.  Must expose `env.success` (bool) after a
        terminated episode — this is set by all LineFollowing / ObstacleAvoidance
        environments in this project.
    retry_on_failure : bool
        If False the wrapper is a transparent pass-through (useful for turning
        the behaviour off via a flag without changing calling code).
    retry_on_timeout : bool
        If True, also replay the same scenario on timeout (truncated episodes).
        Default False — timeouts are ambiguous and replaying them can slow
        convergence by over-weighting hard-but-not-catastrophic layouts.
    max_retries : int
        Maximum consecutive retries of the same scenario before forcing a fresh
        reset.  Prevents the agent getting stuck indefinitely on a pathological
        layout.  Default 5.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        retry_on_failure: bool = True,
        retry_on_timeout: bool = False,
        max_retries: int = 5,
    ):
        super().__init__(env)
        self._retry_on_failure = retry_on_failure
        self._retry_on_timeout = retry_on_timeout
        self._max_retries = max_retries

        self._current_seed: int | None = None
        self._should_retry: bool = False
        self._consecutive_retries: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_seed(self) -> int:
        """Draw a fresh random seed using numpy's default RNG."""
        return int(np.random.randint(0, 2**31 - 1))

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset the environment, optionally replaying the same scenario.

        If a retry is due AND the consecutive-retry limit has not been hit,
        the stored seed is reused (same path + spawn).  Otherwise a new seed
        is generated and stored for future retries.
        """
        if (
            self._retry_on_failure
            and self._should_retry
            and self._current_seed is not None
            and self._consecutive_retries < self._max_retries
        ):
            # --- retry: replay same scenario ---
            use_seed = self._current_seed
            self._consecutive_retries += 1
        else:
            # --- fresh: new random (or caller-supplied) scenario ---
            use_seed = seed if seed is not None else self._new_seed()
            self._current_seed = use_seed
            self._consecutive_retries = 0

        self._should_retry = False
        return super().reset(seed=use_seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            success = bool(getattr(self.unwrapped, "success", False))
            failed = terminated and not success
            timed_out = truncated

            self._should_retry = (
                self._retry_on_failure
                and (
                    (failed)
                    or (self._retry_on_timeout and timed_out)
                )
            )

        return obs, reward, terminated, truncated, info
