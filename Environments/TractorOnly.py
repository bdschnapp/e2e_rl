"""
Tractor-only (NO-TRAILER) lane-following environments.

These mirror the deployed ``lidar_24`` observation pipeline
(``LidarStateObservationLineFollowingEnv`` /
``ReverseLidarStateObservationLineFollowingEnv``) but model a SINGLE vehicle
with no trailer. Every trailer / hitch term is dropped from both the
observation vector and the reward, and the trailer kinematics are not
integrated.

Design (ADD-ONLY — nothing in the existing tree is modified)
------------------------------------------------------------
The existing env hierarchy is reused wholesale by subclassing:

    LidarStateObservationLineFollowingEnv          (forward, 8 + N lidar)
    ReverseLidarStateObservationLineFollowingEnv   (reverse, 8 + N lidar)

A ``TractorOnlyMixin`` does three things on top of the inherited behaviour:

1. **Rigidly aligns the trailer to the tractor** after every reset/step
   (``trailer.yaw = tractor.p``; trailer axle placed directly behind the rear
   axle along the tractor heading). The underlying ``StateSpaceTractorTrailer``
   still *exists* (we don't touch ``VehicleModels``), but because we overwrite
   the trailer pose every tick the trailer dynamics never influence anything:
   the hitch angle is identically 0 and the trailer cross-track/heading errors
   collapse onto the tractor's. This keeps every inherited routine that reads
   ``self.vehicle.trailer`` (collision/jackknife check, reverse lidar mount,
   curvature lookahead, BEV anchor) working without edits.

2. **Reduces the state vector** from the 8-dim trailer layout
   ``[s, γ, e_y, e_ψ, e_y_t, e_ψ_t, κ₁, κ₂]`` to the 5-dim tractor-only layout
   ``[s, e_y, e_ψ, κ₁, κ₂]`` (drops the hitch angle γ and BOTH trailer error
   terms). The lidar beams are appended exactly as before, so:

       obs_dim = 5 + lidar_beams      (e.g. 29 for the lidar_24 deployment)

   vs. the trailer obs_dim = 8 + lidar_beams (= 32 for lidar_24).

3. **Strips trailer/hitch terms from the reward** while preserving the
   structure (progress / path-tracking / speed shaping) of every reward_mode.

Because the trailer is rigidly aligned, e_y_t ≈ e_y and the hitch angle ≈ 0,
so the *behaviour* is well defined even though the inherited code still
references the trailer object; the tractor-only reward simply doesn't reward
or penalise those (now-degenerate) quantities.
"""

import numpy as np
from gymnasium import spaces

import e2erl_utils.config as config
from Environments.ObstacleAvoidance import (
    LidarStateObservationLineFollowingEnv,
    Pose,
    get_obstacle_distances,
)
from Environments.LineFollowing import (
    ReverseLidarStateObservationLineFollowingEnv,
    compute_curvature,
)


# Number of state dims in the tractor-only vector (vs. 8 for tractor+trailer).
# Layout: [s, e_y, e_ψ, κ₁, κ₂]
TRACTOR_ONLY_STATE_DIM = 5


def _make_tractor_only_state_bounds():
    """(low, high) for the 5-dim tractor-only state vector.

    Mirrors LineFollowing._make_state_obs_bounds() but without the hitch-angle
    bound and the two trailer-error bounds.
    """
    low = np.array([
        -config.steering_observation,
        -config.cross_track_distance_observation,
        -config.cross_track_angle_observation,
        -config.curvature_observation,
        -config.curvature_observation,
    ], dtype=np.float32)
    high = np.array([
        config.steering_observation,
        config.cross_track_distance_observation,
        config.cross_track_angle_observation,
        config.curvature_observation,
        config.curvature_observation,
    ], dtype=np.float32)
    return low, high


class TractorOnlyMixin:
    """Shared tractor-only behaviour: rigid trailer alignment, reduced state
    vector, and trailer-free reward. Mixed in BEFORE the concrete lidar env so
    its method overrides take precedence (cooperative super())."""

    # ------------------------------------------------------------------ trailer
    def _align_trailer_to_tractor(self):
        """Place the (ignored) trailer rigidly inline with the tractor so the
        hitch angle is 0 and all trailer-derived quantities collapse onto the
        tractor. Called after reset and after every physics step."""
        v = self.vehicle
        v.trailer.yaw = v.p
        v.trailer.yaw_rate = 0.0
        # Hitch at the tractor rear axle; trailer axle L behind it, all colinear
        # with the tractor heading → trailer sits directly behind the tractor.
        hitch_x = v.x - v.lr * np.cos(v.p)
        hitch_y = v.y - v.lr * np.sin(v.p)
        v.trailer.x = hitch_x - v.trailer.L * np.cos(v.p)
        v.trailer.y = hitch_y - v.trailer.L * np.sin(v.p)

    # ------------------------------------------------------------------- render
    def _render_vehicle(self, surface=None):
        """Tractor-only render: draw ONLY the tractor body + steering indicator.
        The base _render_vehicle also draws the (ignored, rigidly-aligned)
        trailer rectangle, which is misleading for a no-trailer policy — so this
        override reproduces just the tractor parts and omits the trailer."""
        from Environments.TractorTrailer import (
            TRACTOR_LENGTH, TRACTOR_WIDTH, COLOR_TRACTOR,
        )
        canvas = surface if surface is not None else self.canvas
        v = self.vehicle
        self._draw_rotated_rect(
            canvas, v.x, v.y, v.p, TRACTOR_LENGTH, TRACTOR_WIDTH, COLOR_TRACTOR,
        )
        # Steering indicator at the tractor's center-front (mirrors the base).
        indicator_length = TRACTOR_LENGTH * 0.2
        indicator_offset = TRACTOR_LENGTH * 0.25
        base_x = v.x + indicator_offset * np.cos(v.p)
        base_y = v.y + indicator_offset * np.sin(v.p)
        end_x = base_x + indicator_length * np.cos(v.p + v.s)
        end_y = base_y + indicator_length * np.sin(v.p + v.s)
        self._draw_rotated_line(
            canvas, base_x, base_y, end_x, end_y, color=(255, 0, 0), width=2,
        )
        return canvas

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        # super().reset() built the first observation BEFORE we could align the
        # trailer, so realign and rebuild the observation for a clean tractor-
        # only obs (hitch=0, trailer error = tractor error).
        self._align_trailer_to_tractor()
        obs = self._get_obs()
        info = out[1] if isinstance(out, tuple) and len(out) == 2 else {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # The parent step ran the trailer kinematics and built obs/reward off
        # the drifted trailer. Realign and rebuild so the trailer never
        # influences the tractor-only obs (reward already excludes trailer
        # terms via get_reward override below).
        self._align_trailer_to_tractor()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- observation
    def _get_state_vector_obs(self):
        """5-dim tractor-only state [s, e_y, e_ψ, κ₁, κ₂].

        Drops γ (hitch) and the two trailer-error terms vs. the 8-dim base.
        """
        error, error_theta = self.get_vehicle_errors()
        k1 = compute_curvature(self, lookahead_steps=10, max_curvature=config.curvature_observation)
        k2 = compute_curvature(self, lookahead_steps=20, max_curvature=config.curvature_observation)
        return np.array([
            self.vehicle.s, error, error_theta, k1, k2,
        ], dtype=np.float32)

    # ------------------------------------------------------------------ reward
    def get_reward(self, error, error_theta, error_t, error_theta_t):
        """Trailer/hitch-free reward. Same structure per reward_mode as the
        base classes but with every trailer-error and hitch term removed.

        error_t / error_theta_t are accepted (the inherited _get_reward passes
        them) but ignored.
        """
        if self._get_term():
            return self._terminal_reward()

        return self._running_reward(error, error_theta)

    def _terminal_reward(self):
        """Override per direction (forward vs reverse use different scales)."""
        raise NotImplementedError

    def _running_reward(self, error, error_theta):
        """Override per direction."""
        raise NotImplementedError


class TractorOnlyLidarStateLineFollowingEnv(
    TractorOnlyMixin, LidarStateObservationLineFollowingEnv
):
    """FORWARD tractor-only lane-following with state + lidar observation.

    Observation: [s, e_y, e_ψ, κ₁, κ₂, d₀, …, d_{N-1}]  → 5 + lidar_beams dims.

    Mirrors ``LidarStateObservationLineFollowingEnv`` (the forward lidar_24
    deployment env) minus the trailer. Constructor signature matches the base
    so the train_lab_model.py variable-speed __init__ wrapper still applies.
    """

    def __init__(self, render_mode="human", max_episode_steps=1000,
                 lidar_beams=16, reward_mode: str = "dense"):
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            lidar_beams=lidar_beams,
            reward_mode=reward_mode,
        )

        # Rebuild obs space: 5-dim tractor-only state + lidar (base built an
        # 8-dim-state-based space; we shrink the state portion to 5).
        state_low, state_high = _make_tractor_only_state_bounds()
        lidar_low = np.zeros(self.lidar_beams, dtype=np.float32)
        lidar_high = np.ones(self.lidar_beams, dtype=np.float32)
        self.obs_low = np.concatenate([state_low, lidar_low])
        self.obs_high = np.concatenate([state_high, lidar_high])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32
        )
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_obs(self):
        # StateObservationLineFollowingEnv._get_obs returns the (now 5-dim)
        # state via our _get_state_vector_obs override; append forward lidar
        # mounted on the tractor (same as the base forward lidar env).
        state_obs = self._get_state_vector_obs()
        lidar_pose = Pose(
            x=self.vehicle.x,
            y=self.vehicle.y,
            yaw=self.vehicle.p,
        )
        lidar_distances = get_obstacle_distances(
            self.occ_grid,
            lidar_pose,
            num_sensors=self.lidar_beams,
        )
        self.observation = np.concatenate(
            [state_obs, lidar_distances.astype(np.float32)]
        )
        return self.observation

    # ----- forward reward (mirror of LineFollowingEnv.get_reward, no trailer)
    def _terminal_reward(self):
        return 100.0 if self.success else -100.0

    def _running_reward(self, error, error_theta):
        progress_reward = 0.5 * self.vehicle.xd
        tractor_penalty = (error ** 2) + (error_theta ** 2)

        if self.reward_mode in ("dense", "tractor_focus"):
            # Both collapse to tractor-only (the trailer_penalty term in
            # "dense" is dropped); identical here by construction.
            return progress_reward - tractor_penalty - self._proximity_penalty()

        if self.reward_mode == "multiplicative":
            path_term = np.exp(-(abs(error) + 0.5 * abs(error_theta)))
            # Additive speed floor (see LineFollowing.py): a moving vehicle
            # keeps a positive floor so it never prefers crashing into a wall to
            # terminate. The "park on line" exploit is prevented structurally by
            # the explicit STOP action (parking ends the episode, no farming).
            return (
                4.0 * path_term
                + 0.5 * np.clip(self.vehicle.xd, 0.0, 1.0)
                - self._proximity_penalty()
            )

        if self.reward_mode == "guided":
            dense_rew = progress_reward - tractor_penalty
            alpha = max(0.0, 1.0 - self._guide_steps / self.transition_timesteps)
            self._guide_steps += 1
            if alpha == 0.0:
                return dense_rew - self._proximity_penalty()
            pp_action = self._compute_guide_pp_action()
            max_steer_rate = np.deg2rad(config.steering_action)
            steer_diff = (self._last_action[0] - pp_action[0]) / (2.0 * max_steer_rate)
            guide_rew = progress_reward - 5.0 * steer_diff ** 2
            return alpha * guide_rew + (1.0 - alpha) * dense_rew - self._proximity_penalty()

        raise ValueError(f"Unsupported forward reward_mode={self.reward_mode!r}")


class ReverseTractorOnlyLidarStateLineFollowingEnv(
    TractorOnlyMixin, ReverseLidarStateObservationLineFollowingEnv
):
    """REVERSE tractor-only lane-following with state + lidar observation.

    Observation: [s, e_y, e_ψ, κ₁, κ₂, d₀, …, d_{N-1}]  → 5 + lidar_beams dims.

    Mirrors ``ReverseLidarStateObservationLineFollowingEnv`` minus the trailer.
    With no trailer, the lidar (which the base mounts on the trailer, since the
    trailer leads in reverse) is mounted on the tractor's rear, facing the
    direction of travel (tractor heading + π). Constructor signature matches
    the base (incl. fixed_speed) so the variable-speed wrapper still applies.
    """

    def __init__(self, render_mode="human", max_episode_steps=1000,
                 lidar_beams=16, reward_mode: str = "dense",
                 fixed_speed: bool = True):
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            lidar_beams=lidar_beams,
            reward_mode=reward_mode,
            fixed_speed=fixed_speed,
        )

        state_low, state_high = _make_tractor_only_state_bounds()
        lidar_low = np.zeros(self.lidar_beams, dtype=np.float32)
        lidar_high = np.ones(self.lidar_beams, dtype=np.float32)
        self.obs_low = np.concatenate([state_low, lidar_low])
        self.obs_high = np.concatenate([state_high, lidar_high])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32
        )
        self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_obs(self):
        state_obs = self._get_state_vector_obs()
        # No trailer: mount lidar on the tractor rear axle, facing backward
        # (the direction the vehicle travels in reverse), matching how the base
        # reverse env points its (trailer) lidar along the approach direction.
        rear_x = self.vehicle.x - self.vehicle.lr * np.cos(self.vehicle.p)
        rear_y = self.vehicle.y - self.vehicle.lr * np.sin(self.vehicle.p)
        lidar_pose = Pose(
            x=rear_x,
            y=rear_y,
            yaw=self.vehicle.p + np.pi,
        )
        lidar_distances = get_obstacle_distances(
            self.occ_grid,
            lidar_pose,
            num_sensors=self.lidar_beams,
        )
        self.observation = np.concatenate(
            [state_obs, lidar_distances.astype(np.float32)]
        )
        return self.observation

    # ----- reverse reward (mirror of ReverseState...get_reward, no trailer)
    def _terminal_reward(self):
        return 200.0 if self.success else -500.0

    def _running_reward(self, error, error_theta):
        # Path penalty without trailer terms (drops 0.5*e_t² + 0.25*e_θt²).
        path_penalty = error ** 2 + 0.5 * error_theta ** 2
        # Reward slow, controlled reverse (xd < 0 in reverse).
        reverse_reward = 3.0 * np.clip(-self.vehicle.xd, 0.0, 1.0)

        if self.reward_mode in ("dense", "no_hitch"):
            # "no_hitch" already dropped the jackknife term; with no trailer
            # "dense" also has no jackknife/trailer terms, so they coincide.
            return reverse_reward - path_penalty - self._proximity_penalty()

        if self.reward_mode == "multiplicative":
            path_term = np.exp(-(abs(error) + 0.5 * abs(error_theta)))
            # Additive speed floor (reverse uses |xd| via clip(-xd)); see fwd.
            return (
                5.0 * path_term
                + 0.5 * np.clip(-self.vehicle.xd, 0.0, 1.0)
                - self._proximity_penalty()
            )

        if self.reward_mode == "guided":
            dense_rew = reverse_reward - path_penalty
            alpha = max(0.0, 1.0 - self._guide_steps / self.transition_timesteps)
            self._guide_steps += 1
            if alpha == 0.0:
                return dense_rew - self._proximity_penalty()
            pp_action = self._compute_guide_pp_action()
            max_steer_rate = np.deg2rad(config.steering_action)
            steer_diff = (self._last_action[0] - pp_action[0]) / (2.0 * max_steer_rate)
            guide_rew = reverse_reward - 5.0 * steer_diff ** 2
            return alpha * guide_rew + (1.0 - alpha) * dense_rew - self._proximity_penalty()

        raise ValueError(f"Unsupported reverse reward_mode={self.reward_mode!r}")
