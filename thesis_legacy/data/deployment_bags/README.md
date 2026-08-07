# Hardware deployment bags (A-to-B trials, 2026-08-05)

Source: AgileX lab robot, `electrans_robot@agilex:~/Ben/bags/`. All nine A2B
bags recorded that day are pulled here. The same physical lane segment is
driven in both directions; the tractor leads forward, the trailer leads in
reverse.

Four trials completed the maneuver under policy control and are the ones the
thesis reports. Regenerate this screening with
`python thesis/scripts/generate_deployment_results.py --survey`.

| Bag | Dir | Driven | Outcome |
|---|---|---|---|
| `run_A2B_fwd_20260805_100756` | fwd | 23.6 m | **completed (F1)** |
| `run_A2B_fwd_20260805_101802` | fwd | 23.6 m | **completed (F2)** |
| `run_A2B_rev_20260805_104424` | rev | 0 m | stationary bench session, no velocity ever commanded |
| `run_A2B_rev_20260805_110654` | rev | 4.6 m | operator takeover |
| `run_A2B_rev_20260805_112226` | rev | 11.7 m | operator takeover |
| `run_A2B_rev_20260805_115949` | rev | 11.6 m | operator takeover |
| `run_A2B_rev_20260805_120951` | rev | 22.6 m | **completed (R1)** |
| `run_A2B_rev_20260805_124244` | rev | 12.7 m | operator takeover |
| `run_A2B_rev_20260805_124549` | rev | 22.6 m | **completed (R2)** |

Nothing in the bags labels a run as failed. A takeover is identified as a
zero velocity command while the vehicle keeps moving against the commanded
direction, at speeds above the bridge's own ceiling.

## Contents

- `run_*/` — raw rosbag2 (`.db3` + `metadata.yaml`), ~60 MB total.
  Consider gitignoring these; the CSV extracts below are sufficient to
  regenerate every thesis artifact.
- `run_*.log` — `ros2 bag record` console logs.
- `csv/run_*/` — flat per-topic CSV extracts plus `centerlines.npz`.
- `run_*_series.csv` — merged per-control-tick time series with derived
  geometry (trailer pose, cross-track and heading errors, path arclength).
- `deployment_summary.csv` / `.json` — per-run summary statistics.

## Pipeline

1. `thesis/scripts/extract_deployment_bags.py` runs **on the robot** (it needs
   the `autoware_vehicle_msgs` / `autoware_control_msgs` definitions):

   ```bash
   scp thesis/scripts/extract_deployment_bags.py electrans_robot@agilex:/tmp/
   ssh electrans_robot@agilex \
     'source /opt/ros/humble/setup.bash && source ~/Ben/rl_sim_to_real/install/setup.bash;
      python3 /tmp/extract_deployment_bags.py /tmp/bagcsv ~/Ben/bags/run_A2B_*'
   rsync -az electrans_robot@agilex:/tmp/bagcsv/ thesis/data/deployment_bags/csv/
   ```

2. `thesis/scripts/generate_deployment_results.py` runs locally and writes
   `thesis/tables/deployment_runs.tex`,
   `thesis/figures/experiment_results/deployment_trajectories.pdf`, and
   `thesis/figures/experiment_results/deployment_tracking.pdf`.

## Metric conventions

Statistics cover the *policy-controlled segment*: drive enabled in the trial's
direction, and a non-zero velocity command in that direction, closing at the
first zero-velocity command sustained for 1 s. Manual recovery driving after a
takeover is excluded.

Errors are geometric, computed from `/localization/kinematic_state` against the
active `/planning/lane_reference/centerline`, not read back from
`/rl_bridge/state_vector` (which has passed through the Smith predictor and a
frame mirror). The trailer axle is reconstructed from the onboard hitch
estimate with `L_t = 2.8 m` (set by `rl_bridge_node.py`). The reconstruction
was validated against the logged observation vector: |r| >= 0.98 on
articulation, tractor error, and trailer error in all four runs.
