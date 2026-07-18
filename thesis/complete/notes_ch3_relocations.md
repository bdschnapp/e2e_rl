# Passages removed from Ch3 (Modeling, Simulation, and Problem Formulation) for relocation

The Ch3 structural critic pass (2026-07) found that Ch3 had drifted beyond "modeling,
simulation, and problem formulation" into controller-design methodology, experimental
protocol, results framing, and sim-to-real strategy that later chapters own. Those passages
were removed from Ch3 and are preserved here verbatim so they can be adapted into the
chapters noted. Nothing is lost; it is staged for reuse. Spelling is already Canadian.

Destination summary:
- Classical-control (LQR) framing -> Ch5 discussion (implementation already lives in
  `appendices/appendix_classical_controllers.tex`, `sec:app_cc_lqr`).
- Testing-protocol / experimental-validity / experiment-definition tables -> Ch5 (results),
  which is being rewritten and is gated on the in-progress studies.
- Sim-to-Real / ROS2 transition strategy -> Ch6 (`ch:sim_to_real`).

---

## -> Ch5 (Results): classical-control "learned alternative to the model-based chain" framing

Was the tail of Ch3 "Model-Based Control Formulation". The LQR cost functional, gain design,
and block-diagonal structure are already in `app:classical_controllers`; only the framing
below (RL as a learned alternative to the model-based chain) is worth reusing, as Ch5
discussion or Ch1 motivation. A one-line model property ("$(A,B)$ is controllable across the
operating-speed range") was kept in Ch3 Tractor Dynamics.

> This linear design is exact only near its operating point: the pure-pursuit, PID, and MPC
> baselines extend the same model-based philosophy to the full nonlinear path-tracking
> problem, and the deployment's conventional pipeline (Ch6) uses an analogous LQR/MPC
> tracker. The reinforcement-learning controller developed in this thesis is, in effect, a
> learned alternative to this model-based chain that requires neither an explicit
> linearization nor hand-tuned gains.

---

## -> Ch5 (Results): testing protocol, metrics, and experimental-validity framing

Was Ch3 "Testing Protocols" intro. This is experimental methodology / results-chapter
framing, not problem formulation.

> The experimental workflow is scriptable end-to-end. Scenario generation,
> observation-modality training, trained-policy evaluation, controller tuning, and
> multi-controller benchmarking are each exposed through standalone scripts. The
> benchmarking pipeline runs all controllers on the same pre-generated scenario set and logs
> per-step metrics including tractor and trailer cross-track error, hitch angle, completion,
> collision, jackknife events, obstacle clearance, and inference latency.
>
> The experiments are treated as two related but distinct problem classes:
> 1. **Path-tracking control**: forward and reverse lane following without obstacles. In this
>    setting all methods are solving the same task at a common fixed speed, so TD3 can be
>    compared directly against pure-pursuit, PID, and MPC as lateral/path-tracking controllers.
> 2. **Integrated navigation**: obstacle-avoidance tasks in which the policy must implicitly
>    combine local planning and control. In this setting a direct comparison against
>    controller-only baselines is not fair unless those baselines are also given a local planner.
>
> This distinction is important for experimental validity. A controller that only tracks a
> supplied path should not be evaluated against an RL policy that must both choose a
> collision-free local path and execute it, unless the classical baseline is paired with an
> equivalent planner.

Also the detailed fixed-vs-modulated-speed evaluation protocol (was in Ch3 "Action Space");
a one-sentence version was kept in the Ch3 MDP, the full protocol goes to Ch5:

> For evaluation, this full action space is not used identically in every study. In the
> no-obstacle line-following experiments, the longitudinal speed is held at a fixed operating
> value ($5$ m/s forward, $-5$ m/s in reverse) so that all methods are compared on the same
> path-tracking problem, leaving the agent to command only the steering rate. In the
> obstacle-avoidance experiments, speed modulation remains part of the task because the agent
> must trade progress against collision risk and path geometry.

---

## -> Ch5 (Results): lane-following experiment definitions table (was tab:forward_experiments)

| Experiment | Objective | Success Metric |
|---|---|---|
| Path Fidelity (Tractor) | Maintain low CTE for tractor on varying curvature paths | Mean Absolute CTE |
| Path Fidelity (Trailer) | Maintain low CTE for trailer, ensuring full vehicle stays in lane | Mean Absolute CTE |
| Hitch Stability | Keep articulation angle within safe bounds | Max $|\gamma|$ per episode |

---

## -> Ch5 (Results): obstacle-avoidance comparison strategy

Was Ch3 "Testing Protocols > Obstacle Avoidance".

> The obstacle avoidance task tests the agent's ability to navigate through a lane containing
> static obstacles. Success is measured by collision rate and the agent's ability to complete
> the lane while maintaining hitch stability.
>
> Obstacle experiments are framed as a secondary study rather than merged into the primary
> controller benchmark. Two defensible comparison strategies are available:
> - **Planner-controlled comparison**: provide the classical baselines with the same
>   obstacle-aware local path and evaluate only the downstream tracking problem; or
> - **End-to-end navigation comparison**: compare the RL obstacle policy against complete
>   planner+controller stacks, not against controllers in isolation.
>
> In the present study, obstacle avoidance is treated as an RL feasibility extension and the
> main quantitative comparison is reserved for the no-obstacle forward and reverse tracking
> tasks. Under this structure, no-obstacle experiments use fixed speed for both learned and
> classical methods, whereas obstacle experiments allow the RL agent to command speed as part
> of the end-to-end navigation problem.

---

## -> Ch6 (Sim-to-Real): ROS2 transition strategy

Was Ch3 "Testing Protocols > Sim-to-Real and ROS2 Transition Strategy". Belongs with the
deployment chapter. Integrate deliberately (read Ch6 first to avoid redundancy with the
existing deployment description).

> The learning experiments in this thesis are conducted in simulation, but the software
> architecture was designed from the outset to port onto a ROS2-based robotics stack --- a
> port that has since been realized on physical hardware (Ch6). The key design decision is
> that the policy and controller interfaces are expressed in terms of generic observations and
> low-level control commands rather than simulator-specific rendering calls. Each control
> method consumes either a vector observation, a lidar-style range vector, or a BEV image
> paired with state variables, and produces steering-rate and speed commands. This abstraction
> is directly compatible with a ROS2 node graph in which sensor drivers publish observations, a
> policy node performs inference, and a downstream interface node converts the commands into
> actuator messages.
>
> For no-obstacle path-tracking experiments, the fixed-speed formulation used in the thesis
> also improves deployment clarity: a real vehicle implementation can treat longitudinal speed
> regulation as a separate closed-loop subsystem while the learned or analytical controller
> handles lateral and articulation control. For obstacle-navigation experiments, the policy
> retains speed authority because longitudinal modulation is part of the task definition. This
> separation reduces ambiguity when mapping the thesis experiments onto a real robotic
> architecture.
>
> The ROS2 transition therefore reduces to replacing only the environment-facing components:
> - the simulated state vector with subscribed vehicle-state estimates;
> - the simulated lidar or BEV renderer with ROS2 sensor topics or perception outputs;
> - the Pygame environment step call with a command publisher targeting steering and speed interfaces;
> - the episode reset logic with scenario initialization and safety supervision on the robot.
>
> The policy network, observation formatting, benchmark metrics, and controller-selection logic
> can remain largely unchanged. This is important because it turns ROS2 deployment from a full
> reimplementation problem into an interface-substitution problem.
>
> However, portability should not be overstated. To close the remaining gap, the simulator must
> be augmented with domain randomization and system identification so that the learned policy is
> exposed to uncertainty in tire parameters, friction, mass distribution, actuator delay, sensor
> noise, and hitch compliance. Several of these factors are taken up directly in the deployment
> of Ch6 --- notably actuator delay, which is addressed there by a Smith-predictor compensator
> --- but in the absence of systematic randomization, strong simulation performance alone is not
> evidence of hardware readiness.

---

## -> Ch4 (Learning Framework): observation-modality detail stripped from Ch3 tasks

The Ch3 Environment Tasks section previously catalogued the vector/lidar/image observation
heads and noted reverse BEV/lidar observation classes and BEV obstacle-avoidance variants.
Ch4 owns observation design, so Ch3 now points to Ch4. The specific points worth ensuring
Ch4 (or Ch5) states: reverse tasks have explicit reverse BEV and reverse lidar observation
classes; obstacle tasks have BEV variants for both directions, enabling a range-based vs
image-based perception comparison within the same local-planning task.
