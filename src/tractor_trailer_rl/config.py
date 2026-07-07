"""Explicit configuration for tractor_trailer_rl.

Replaces e2e_rl's ``e2erl_utils.config`` module-of-globals (which forced callers
to monkeypatch values BEFORE importing the env modules, because TractorTrailer.py
captured 7 of them into module globals at import time). Here every value lives in
a frozen dataclass passed to constructors / the observation builder, so there is
nothing global to patch.

Two presets:
  * ``truck_config()`` — the original e2e_rl semi-truck-scale defaults (used as the
    parity oracle).
  * ``lab_config()``   — the AgileX 1/8-scale lab values (what the sim-to-real
    pipeline previously injected via ``_apply_lab_config_overrides``).

Parity note: the default field values below equal the e2e_rl defaults, so a
``Config()`` built from ``truck_config()`` reproduces upstream behaviour. The
speed-scaled lookahead improvement is OFF by default (``LookaheadMode.FIXED_SAMPLES``)
so parity holds until it is explicitly switched on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math


class Direction(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class VehicleKind(Enum):
    TRAILER = "trailer"
    TRACTOR_ONLY = "tractor_only"


class ActionMode(Enum):
    FIXED_SPEED = "fixed_speed"       # 1-D [steer_rate]
    VARIABLE_SPEED = "variable_speed"  # 2-D [steer_rate, velocity]
    STOP_SIGNAL = "stop_signal"        # 2-D [steer_rate, stop]; constant speed
    DISCRETE = "discrete"              # Discrete(n_steer_bins*2): steer bin x {go,stop}


class LookaheadMode(Enum):
    FIXED_SAMPLES = "fixed_samples"    # parity: k1=10, k2=20 path samples ahead
    SPEED_SCALED = "speed_scaled"      # improvement: lookahead distance = v * preview_time_s


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleConfig:
    # Dynamics (StateSpaceVehicleModel)
    m: float = 1500.0
    Iz: float = 3000.0
    Cf: float = 80000.0
    Cr: float = 80000.0
    lf: float = 1.2          # CG -> front axle (m); lab 0.33
    lr: float = 1.6          # CG -> rear axle (m);  lab 0.32
    Cd: float = 0.208
    A: float = 2.4
    rho: float = 1.225
    dt: float = 0.1
    # Longitudinal PID gains (were hardcoded in BasicVehicleModel)
    kp: float = 3.0
    ki: float = 0.3
    kd: float = 0.05
    # Body geometry (rendering + lidar vehicle half-width)
    tractor_length_m: float = 4.5   # lab 1.0
    tractor_width_m: float = 2.5    # lab 0.65
    trailer_length_m: float = 10.0  # == kinematic trailer wheelbase L; lab 2.8
    trailer_width_m: float = 2.5    # lab 0.65
    # First-class actuator lag (was the _patch_actuator_lag monkeypatch).
    # 0.0 => delay-free (parity with training). The Smith predictor's
    # delayed shadow sets these to the ROS time constants.
    steer_tau_s: float = 0.0
    velocity_tau_s: float = 0.0
    # Steering-angle clamp (rad). Default pi/4 (parity with e2e_rl). A shunt truck
    # steers to ~0.87 rad (50 deg) for tight yard maneuvering. Used by the batched
    # env; the scalar bicycle keeps pi/4 for oracle parity.
    max_steer_angle_rad: float = math.pi / 4

    @property
    def vehicle_args(self) -> dict:
        """Dict in the shape StateSpaceVehicleModel.__init__ expects."""
        return dict(m=self.m, Iz=self.Iz, Cf=self.Cf, Cr=self.Cr,
                    lf=self.lf, lr=self.lr, Cd=self.Cd, A=self.A, dt=self.dt)


@dataclass(frozen=True)
class WorldConfig:
    # Canvas / world extent (world_w = width_px * meters_per_pixel). Default 150x90 m.
    width_px: int = 1500
    height_px: int = 900
    meters_per_pixel: float = 0.1
    # Occupancy grid + lane corridor
    grid_res_m: float = 0.10
    lane_centerline_half_width_m: float = 5.0   # lab 1.41
    lane_shoulder_m: float = 0.50               # lab 0.20
    lane_sample_ds_m: float = 0.25

    @property
    def width_m(self) -> float:
        return self.width_px * self.meters_per_pixel

    @property
    def height_m(self) -> float:
        return self.height_px * self.meters_per_pixel

    @property
    def corridor_half_m(self) -> float:
        return self.lane_centerline_half_width_m + self.lane_shoulder_m


@dataclass(frozen=True)
class ObsConfig:
    lidar_beams: int = 24
    lidar_fov_deg: float = 120.0
    lidar_range_m: float = 20.0
    lidar_step_m: float = 0.05
    # Observation-space bounds (for the gym Box; not used by the raw obs values)
    steering_observation: float = math.pi / 6
    hitch_angle_observation: float = math.pi / 2
    cross_track_distance_observation: float = 100.0
    cross_track_angle_observation: float = math.pi / 4
    curvature_observation: float = 0.3
    error_theta_scale: float = 1.0
    # Bird's-eye-view (BEV) occupancy image for the Stage-2 perception ablation.
    # bev_size=0 disables it (default => vector/lidar obs unchanged). When >0 the
    # observation becomes Dict{vector, image=(1,S,S)}, the image rendered analytically
    # in the vehicle frame (batched/bev.py) — no pygame, GPU-parallel. bev_range_m is
    # the half-extent (metres) the S x S grid spans around the anchor pose.
    bev_size: int = 0
    bev_range_m: float = 20.0


@dataclass(frozen=True)
class LookaheadConfig:
    mode: LookaheadMode = LookaheadMode.FIXED_SAMPLES
    # FIXED_SAMPLES (parity): curvature read k1/k2 samples ahead on the path.
    k1_samples: int = 10
    k2_samples: int = 20
    # SPEED_SCALED (improvement): lookahead distance = clip(|v|*T, dist_min, dist_max).
    preview_time_s: tuple[float, float] = (1.5, 3.0)
    dist_min_m: float = 1.0
    dist_max_m: float = 30.0
    # Heading-tangent preview time (0.0 => use the nearest+1 neighbour, parity).
    heading_preview_s: float = 0.0
    max_curvature: float = 0.3


@dataclass(frozen=True)
class ActionConfig:
    mode: ActionMode = ActionMode.FIXED_SPEED
    steering_action_deg: float = 25.0   # max steer-rate magnitude
    fixed_speed_m_s: float = 5.0        # = initial_xd
    # variable_speed bounds (reverse mirrors to [-v_max, -v_min])
    v_min: float = 0.5
    v_max: float = 3.0
    # stop_signal
    stop_threshold: float = 0.5
    stop_penalty: float = 200.0
    stop_noise_sigma: float = 0.1
    steer_noise_sigma: float = 0.05
    # discrete: number of evenly-spaced steer-rate bins in [-max, +max] (odd => includes 0).
    # 7 => {-max, -2/3, -1/3, 0, +1/3, +2/3, +max}; crossed with 2 speed options {go, stop}.
    n_steer_bins: int = 7


@dataclass(frozen=True)
class PathConfig:
    spacing_m: float = 1.0              # parity (e2e_rl used np.arange(x0, x_end, 1))
    vert_offset_m: float = 45.0
    x_start_m: float = 5.0
    x_end_margin_m: float = 5.0
    mild_only: bool = False
    # Difficulty knob for a curvature curriculum: multiplies the bend AMPLITUDE of
    # lab_seam / lab_corner (peak curvature scales with it too). 1.0 = parity (the
    # trained geometry); <1.0 makes "slightly sharper than gentle" corners so a
    # mild-only base can be ramped onto real corners without the distribution jump
    # that diverges TD3. Applied as a plain multiply on the drawn dy — no extra RNG
    # draw, so curve_scale=1.0 reproduces paths bit-for-bit.
    curve_scale: float = 1.0
    # Scales the sharp/winding bend geometry (width AND amplitude together, so the
    # turn RADIUS scales ~linearly) to match the vehicle. The hardcoded bends are
    # small-vehicle scale (~2.5 m radius); a full-size shunt truck (2.95 m wheelbase,
    # 12.5 m trailer) needs ~12-20 m radii, so shunt_truck_config sets bend_scale~5.
    bend_scale: float = 1.0
    # lab_chicane geometry: y = sign * [ P*tanh(N*(x-c1)) if x<mid else -P*tanh(N*(x-c2)) ]
    # a committed out-and-back bump: two turns `chicane_turn_spacing_m` apart with a
    # long flat middle so a long rig can't shortcut both knees. N sets sharpness
    # (min radius ~ 1/(P*N^2)); the centreline is deliberately NOT perfectly trackable
    # through the turn -- pair with the additive 'corridor' reward. P scaled by curve_scale.
    chicane_amplitude_m: float = 5.0     # P
    chicane_slope: float = 3.0           # N
    chicane_turn_spacing_m: float = 10.0
    # v18 mixture (probabilities over kinds). mild_only -> straight/gentle only.
    # Unlisted kinds (e.g. lab_chicane) default to 0.0; the vector is renormalised.
    kind_probs: dict = field(default_factory=lambda: {
        "straight": 0.18, "gentle": 0.12, "sharp": 0.15,
        "winding": 0.18, "lab_seam": 0.12, "lab_corner": 0.25,
    })
    # SAFE POOL: if set, the env loads a pre-validated path pool from this .npz
    # (keys: xs, ys, lo, hi) instead of generating one, and asserts xs matches the
    # cfg-derived x grid. The pool is built by scripts/build_safe_pool.py, which
    # keeps only paths a pure-pursuit controller completes both forward AND reverse
    # (so no policy can fail merely from an infeasible/"unlucky" spawn geometry).
    # None => generate a fresh random pool as before (default; no behaviour change).
    pool_file: str | None = None


@dataclass(frozen=True)
class SpeedRandomConfig:
    enabled: bool = True
    explicit_min: float | None = None   # if both set, override all path kinds
    explicit_max: float | None = None
    per_kind: dict = field(default_factory=lambda: {
        "straight": (1.0, 5.0), "gentle": (1.0, 5.0),
        "sharp": (0.5, 2.0), "winding": (0.5, 2.0),
        "lab_seam": (0.5, 2.0), "lab_corner": (0.4, 1.5),
    })


@dataclass(frozen=True)
class SpawnConfig:
    """Spawn-pose randomization for the recovery curriculum. All defaults 0.0 =>
    spawn exactly on-path, aligned, hitch=0 (parity). Each value is the half-range
    of a symmetric uniform perturbation applied at reset():
      * lateral_offset_m   — shift perpendicular to the path tangent (cross-track).
      * heading_offset_rad — rotate the tractor heading off the path tangent.
      * hitch_offset_rad   — open the trailer's hitch angle (tractor_yaw - trailer_yaw).
    Keep lateral < corridor half-width and hitch < pi/2 or the episode terminates
    on step 1 (collision / jackknife)."""
    lateral_offset_m: float = 0.0
    heading_offset_rad: float = 0.0
    hitch_offset_rad: float = 0.0


@dataclass(frozen=True)
class TargetSpeedConfig:
    gamma: float = 4.0
    sigma_low: float = 0.2
    sigma_high: float = 0.5
    v_target_min: float = 0.6
    v_target_max: float = 2.0


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "multiplicative"        # dense | tractor_focus | multiplicative | guided | no_hitch
    # guided reward: env-steps over which the PP-imitation weight alpha decays 1->0
    # (then it is pure multiplicative). <=0 => NO decay: alpha pinned at 1.0, so guided
    # stays a pure PP-imitation reward the whole run (Phase-1 "clone PP", fine-tune later).
    guide_transition_steps: int = 100_000
    path_tightness: float = 1.0         # parity 1.0 (lab training used 2.0)
    multiplicative_floor: bool = True   # fix B: additive +0.5*clip(|xd|) floor
    curvature_speed_weight: float = 0.0  # beta; 0 disables curvature speed penalty
    curv_penalty_K: int = 10
    curv_penalty_Kback: int = 5
    target_speed: TargetSpeedConfig | None = None  # None disables (parity)
    # Terminal reward overrides (None => direction default: reverse +200/-500,
    # forward +100/-100). Set terminal_fail MORE negative than ActionConfig.
    # stop_penalty so crashing/jackknifing is penalised WORSE than a deliberate
    # stop — otherwise (equal penalties) the policy fires the stop to escape an
    # impending jackknife at no extra cost ("suicidal stop"). Desired ordering:
    # crash (worst) < stop < drive-well/success (best).
    terminal_success: float | None = None
    terminal_fail: float | None = None


@dataclass(frozen=True)
class ObstacleConfig:
    """In-lane obstacle avoidance + binary stop-gate (Track D).

    ``None`` on ``Config`` (the default) = the pure lane-following env, so the
    reward-shaping / algorithm ablation is completely unaffected. Set this (via
    ``shunt_truck_obstacle_config``) to switch the env factory to the obstacle
    subclass. Obstacles are circles placed near the centreline; the agent sees
    them only through lidar (never the planned avoidance path -> the "hidden local
    planner" reward), and can fire the STOP_SIGNAL action, which is rewarded iff
    the layout difficulty is high and penalised when it is low.
    """
    enabled: bool = True
    max_obstacles: int = 2              # one event cluster per episode (main + optional pair)
    start_margin_m: float = 15.0        # keep clear near spawn (the decision distance)
    goal_margin_m: float = 15.0         # keep clear near the goal band
    influence_radius_m: float = 8.0     # planner longitudinal influence window
    difficulty_bins: int = 61           # lateral bins for the free-gap scan
    # Each episode draws one layout CATEGORY, so the difficulty distribution is
    # controlled and stratifiable (clear/shift = drive-through/around; squeeze =
    # hard-but-passable; blocked = impassable -> stopping is correct). Renormalised.
    layout_probs: dict = field(default_factory=lambda: {
        "clear": 0.30, "shift": 0.25, "squeeze": 0.20, "blocked": 0.25,
    })
    # stop-gate reward, keyed on layout difficulty (position-robust, unlike a
    # progress-scaled bonus). exp_scale(d) = (e^{k d}-1)/(e^k-1) in [0,1].
    stop_difficulty_k: float = 3.0
    stop_hard_reward: float = 60.0      # stop on an impassable layout (must beat a crash)
    stop_easy_penalty: float = 60.0     # stop on an easy layout (must lose to driving on)
    stop_progress_bonus: float = 20.0   # small credit for reaching the obstacle first
    slow_reward_scale: float = 0.1      # difficulty * (1 - norm_speed) * scale
    reward_mode: str = "dense"          # hidden-planner tracking reward (proven forward)


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    direction: Direction = Direction.FORWARD
    vehicle_kind: VehicleKind = VehicleKind.TRAILER
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    lookahead: LookaheadConfig = field(default_factory=LookaheadConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    path: PathConfig = field(default_factory=PathConfig)
    speed_random: SpeedRandomConfig = field(default_factory=SpeedRandomConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obstacle: ObstacleConfig | None = None   # None => pure lane-following (ablation) env
    max_episode_steps: int = 1000
    seed: int | None = None

    @property
    def is_reverse(self) -> bool:
        return self.direction == Direction.REVERSE

    @property
    def is_tractor_only(self) -> bool:
        return self.vehicle_kind == VehicleKind.TRACTOR_ONLY

    def evolve(self, **changes) -> "Config":
        """Return a copy with top-level fields replaced (use dataclasses.replace
        on the sub-configs for nested changes)."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def truck_config(direction: Direction = Direction.FORWARD,
                 vehicle_kind: VehicleKind = VehicleKind.TRAILER,
                 **overrides) -> Config:
    """Original e2e_rl semi-truck defaults (parity oracle)."""
    return Config(direction=direction, vehicle_kind=vehicle_kind, **overrides)


def lab_config(direction: Direction = Direction.FORWARD,
               vehicle_kind: VehicleKind = VehicleKind.TRAILER,
               **overrides) -> Config:
    """AgileX 1/8-scale lab values (was _apply_lab_config_overrides +
    _patch_env_vehicle_params). trailer_length_m=2.8 is the real AgileX trailer
    (single source of truth, see simulator_model.param.yaml)."""
    vehicle = VehicleConfig(
        lf=0.33, lr=0.32,
        tractor_length_m=1.0, tractor_width_m=0.65,
        trailer_length_m=2.8, trailer_width_m=0.65,
    )
    world = WorldConfig(
        lane_centerline_half_width_m=1.41,
        lane_shoulder_m=0.20,
    )
    return Config(direction=direction, vehicle_kind=vehicle_kind,
                  vehicle=vehicle, world=world, **overrides)


def shunt_truck_config(direction: Direction = Direction.FORWARD,
                       vehicle_kind: VehicleKind = VehicleKind.TRAILER,
                       loaded: bool = False,
                       **overrides) -> Config:
    """Real full-size shunt truck (Kalmar Ottawa T2 diesel) + kinematic 53' trailer.

    Replaces the ungrounded 'Tesla Model S' placeholder for the thesis ablation
    (see thesis/shunt_truck_parameters_research.md). Dynamic params describe the
    BOBTAIL TRACTOR body; the trailer is a kinematic hitch link (its mass/load
    transfer is not modelled). ``loaded=True`` approximates a loaded trailer by
    folding the fifth-wheel load into the tractor dynamics (rear-biased CG, higher
    Iz/Cr) — a coarse secondary/robustness variant, NOT the primary vehicle.
    """
    if loaded:
        # ~GCW 81,000 lb combination folded onto the tractor dynamics (approximate).
        vehicle = VehicleConfig(
            m=36740.0, Iz=93000.0, Cf=200000.0, Cr=640000.0,
            lf=1.9, lr=1.05,   # CG shifts rearward under kingpin load (lr shrinks)
            Cd=0.8, A=7.0,
            tractor_length_m=5.11, tractor_width_m=2.54,
            trailer_length_m=12.5, trailer_width_m=2.6,
            max_steer_angle_rad=0.87,
        )
    else:
        # Bobtail diesel T2 — primary-sourced mass/wheelbase, estimated Iz/Cf/Cr.
        vehicle = VehicleConfig(
            m=6577.0, Iz=16000.0, Cf=190000.0, Cr=200000.0,
            lf=1.33, lr=1.62, Cd=0.8, A=7.0,
            tractor_length_m=5.11, tractor_width_m=2.54,
            trailer_length_m=12.5, trailer_width_m=2.6,
            max_steer_angle_rad=0.87,   # 50 deg
        )
    # Full-size path mix: straight/gentle (easy, Stage-1) + sharp (hard, curriculum).
    # winding and the lab_* kinds are small-vehicle geometries that don't scale to a
    # 12.5 m trailer, so they're excluded. bend_scale=4.5 puts 'sharp' at ~18-22 m
    # radius (min ~15 m, above the ~13 m trailer jackknife limit) — tight-but-
    # achievable full-size yard turns; 'gentle' stays ~34 m.
    # spacing_m=0.25 (not the e2e_rl default 1.0): with 1 m path samples the
    # nearest-POINT cross-track error carries a ~0.5 m period-2 artifact (half the
    # spacing) that robust controllers/RL tolerate but high-gain optimal controllers
    # (LQR, aggressive MPC) chase into instability. Finer sampling shrinks it ~4x and
    # makes the optimal-control baselines well-posed; also cleans the RL observation.
    path = PathConfig(bend_scale=4.5, spacing_m=0.25,
                      kind_probs={"straight": 0.2, "gentle": 0.4, "sharp": 0.4})
    # Terminal-reward ordering (match the sibling repos, and the RewardConfig comment):
    # crash (worst) < intentional stop < success. The default forward terminal_fail
    # (-100) is LESS negative than the stop penalty, which inverts the ordering and makes
    # a struggling policy prefer to crash rather than stop once the stop action is live
    # (2-D). Fix: crash = -500 (both directions), stop = -300. The tracking reward is
    # unchanged, so the 1-D-validated stabilization behaviour is preserved; only the
    # (previously dormant, mis-set) stop penalty is corrected.
    reward = RewardConfig(terminal_fail=-500.0)
    # Stop must fire only when clearly intended: high threshold (0.75 of the [-1,1]
    # stop channel) + low stop-action noise (0.05). A prior run used threshold 0.0 +
    # noise 0.5, which tripped the stop constantly. steer_noise_sigma stays 0.05.
    action = ActionConfig(stop_penalty=300.0, stop_threshold=0.75, stop_noise_sigma=0.05)
    return Config(direction=direction, vehicle_kind=vehicle_kind,
                  vehicle=vehicle, path=path, reward=reward, action=action, **overrides)


def shunt_truck_obstacle_config(direction: Direction = Direction.FORWARD,
                                vehicle_kind: VehicleKind = VehicleKind.TRAILER,
                                **overrides) -> Config:
    """Obstacle-avoidance + binary stop-gate variant of the shunt truck (Track D).

    Same vehicle / world / path machinery as ``shunt_truck_config`` (so a
    lane-following policy warm-starts cleanly and the two share the parity-tested
    dynamics), but: (a) ``obstacle`` is populated -> the env factory builds the
    obstacle subclass; (b) the action mode is STOP_SIGNAL so the policy can stop;
    (c) the base reward mode is 'dense' (the obstacle env tracks the *hidden* local
    planner path in dense mode -- the formulation that worked well forward in
    e2e_rl). Paths lean straight/gentle so the obstacle geometry (not the corner)
    is what makes a layout hard. This is a SEPARATE preset -- the lane-following
    ablation presets (obstacle=None) are unchanged and still run as before.
    """
    base = shunt_truck_config(direction=direction, vehicle_kind=vehicle_kind)
    action = replace(base.action, mode=ActionMode.STOP_SIGNAL,
                     stop_threshold=0.5, stop_noise_sigma=0.1)
    reward = replace(base.reward, mode="dense")
    # obstacle-focused path mix: mostly straight/gentle so difficulty comes from
    # the obstacles rather than the corner geometry.
    path = replace(base.path, kind_probs={"straight": 0.5, "gentle": 0.4, "sharp": 0.1})
    return Config(direction=direction, vehicle_kind=vehicle_kind,
                  vehicle=base.vehicle, world=base.world, path=path,
                  action=action, reward=reward, obstacle=ObstacleConfig(),
                  **overrides)


def lab_chicane_config(direction: Direction = Direction.REVERSE,
                       vehicle_kind: VehicleKind = VehicleKind.TRAILER,
                       **overrides) -> Config:
    """Small (20x20 m) lab world with the tanh chicane path + additive 'corridor'
    reward. Same AgileX vehicle/corridor as lab_config so a lab_config-trained
    policy warm-starts cleanly (obs are local/scale-free). The compact world makes
    the whole corner reachable in one reverse episode (no wasted straight), and the
    goal (x > 0.85*W = 17 m) is attainable. Path sampled at 0.25 m so the sharp
    curvature isn't aliased; FIXED_SAMPLES k1/k2 => 2.5 m / 5 m lookahead."""
    vehicle = VehicleConfig(
        lf=0.33, lr=0.32,
        tractor_length_m=1.0, tractor_width_m=0.65,
        trailer_length_m=2.8, trailer_width_m=0.65,
    )
    world = WorldConfig(
        width_px=560, height_px=400, meters_per_pixel=0.05, grid_res_m=0.05,
        lane_centerline_half_width_m=1.41, lane_shoulder_m=0.20,
    )   # 28 m wide x 20 m tall: fits entry setup + turn1 + 10 m middle + turn2 + exit
    path = PathConfig(
        spacing_m=0.25, x_start_m=1.0, x_end_margin_m=1.0, vert_offset_m=10.0,
        kind_probs={"lab_chicane": 1.0},
    )
    return Config(direction=direction, vehicle_kind=vehicle_kind,
                  vehicle=vehicle, world=world, path=path,
                  max_episode_steps=900, **overrides)
