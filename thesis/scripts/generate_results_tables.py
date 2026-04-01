"""Generate thesis LaTeX result tables from experiment CSV outputs.

This script consumes:
  - RL evaluation aggregates at results/<scenario>/<obs_tag>/<reward>/aggregate.csv
  - Benchmark summaries at results/**/summary_<task>.csv

and rewrites the thesis table files in thesis/tables/.

Missing inputs are tolerated: unavailable entries are rendered as '---' so the
thesis still compiles while experiments are in progress.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OBS_ENCODER_ORDER = [
    "state",
    "lidar",
    "bev",
    "bev_ae_frozen",
    "bev_ae_unfrozen",
    "bev_unet_frozen",
    "bev_unet_unfrozen",
]


@dataclass
class Config:
    results_dir: Path
    tables_dir: Path
    obs_lidar_tag: str
    obstacle_lidar_tag: str
    best_obs_forward: str
    best_obs_reverse: str
    best_reward_forward: str
    best_reward_reverse: str


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_last_row(path: Path) -> dict[str, str] | None:
    rows = _read_csv_rows(path)
    return rows[-1] if rows else None


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return None if math.isnan(parsed) else parsed


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}"


def _fmt_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "---"
    return f"{value * 100:.{digits}f}"


def _fmt_degrees(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "---"
    return f"{math.degrees(value):.{digits}f}"


def _fmt_latency(value: float | None) -> str:
    if value is None:
        return "---"
    if value < 1.0:
        return "$<$1"
    return f"{value:.2f}"


def _parse_boolish(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return _parse_float(text)


def _lidar_display_name(obs_tag: str) -> str:
    if obs_tag == "lidar":
        return "Lidar-16"
    if obs_tag.startswith("lidar_"):
        return f"Lidar-{obs_tag.split('_', 1)[1]}"
    return obs_tag


def _label_for_obs_tag(obs_tag: str, lidar_label: str) -> str:
    mapping = {
        "state": "State (MLP)",
        lidar_label: f"{_lidar_display_name(lidar_label)} (MLP)",
        "bev": "BEV CNN (scratch)",
        "bev_ae_frozen": "BEV AE (frozen)",
        "bev_ae_unfrozen": "BEV AE (unfrozen)",
        "bev_unet_frozen": "BEV UNet (frozen)",
        "bev_unet_unfrozen": "BEV UNet (unfrozen)",
    }
    return mapping.get(obs_tag, obs_tag.replace("_", " "))


def _humanize_obs_tag(obs_tag: str) -> str:
    if obs_tag == "state":
        return "state"
    if obs_tag == "lidar":
        return "lidar-16"
    if obs_tag.startswith("lidar_"):
        return f"lidar-{obs_tag.split('_', 1)[1]}"
    if obs_tag == "bev":
        return "BEV CNN (scratch)"
    if obs_tag == "bev_ae_frozen":
        return "BEV AE (frozen)"
    if obs_tag == "bev_ae_unfrozen":
        return "BEV AE (unfrozen)"
    if obs_tag == "bev_unet_frozen":
        return "BEV UNet (frozen)"
    if obs_tag == "bev_unet_unfrozen":
        return "BEV UNet (unfrozen)"
    return obs_tag.replace("_", " ")


def _humanize_reward(reward: str) -> str:
    mapping = {
        "dense": "dense",
        "tractor_focus": "tractor-focus",
        "multiplicative": "multiplicative",
        "guided": "guided",
        "no_hitch": "no-hitch",
    }
    return mapping.get(reward, reward.replace("_", "-"))


def _find_latest(paths: Iterable[Path]) -> Path | None:
    candidates = list(paths)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_eval_aggregates(results_dir: Path) -> dict[str, dict[str, str]]:
    aggregates: dict[str, dict[str, str]] = {}
    for path in sorted(results_dir.glob("**/aggregate.csv")):
        row = _read_last_row(path)
        if not row:
            continue
        label = row.get("label")
        if not label:
            label = str(path.parent.relative_to(results_dir))
        aggregates[label] = row
    return aggregates


def _load_benchmark_rows(results_dir: Path, task: str) -> list[dict[str, str]]:
    path = _find_latest(results_dir.glob(f"**/summary_{task}.csv"))
    if not path:
        return []
    return _read_csv_rows(path)


def _table_lines(
    table_env: str,
    caption: str,
    label: str,
    col_spec: str,
    header_lines: list[str],
    body_rows: list[str],
) -> str:
    lines = [
        f"\\begin{{{table_env}}}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        *header_lines,
        "\\midrule",
        *body_rows,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\end{{{table_env}}}",
        "",
    ]
    return "\n".join(lines)


def _eval_row(aggregates: dict[str, dict[str, str]], label: str) -> dict[str, str] | None:
    return aggregates.get(label)


def _metric(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    return _parse_float(row.get(key))


def _generate_obs_ablation_table(aggregates: dict[str, dict[str, str]], scenario: str, cfg: Config) -> str:
    lidar_label = cfg.obs_lidar_tag
    scenario_title = "Forward" if scenario == "forward" else "Reverse"
    table_label = "tab:obs_ablation_forward" if scenario == "forward" else "tab:obs_ablation_reverse"
    lines: list[str] = []

    for obs_tag in OBS_ENCODER_ORDER:
        resolved_tag = lidar_label if obs_tag == "lidar" else obs_tag
        row = _eval_row(aggregates, f"{scenario}/{resolved_tag}/dense")
        values = [
            _label_for_obs_tag(resolved_tag, lidar_label),
            _fmt_float(_metric(row, "mean_abs_cte_tractor")),
            _fmt_float(_metric(row, "mean_abs_cte_trailer")),
            _fmt_degrees(_metric(row, "max_abs_hitch_angle")),
            _fmt_percent(_metric(row, "completed")),
            _fmt_percent(_metric(row, "jackknifed")),
            _fmt_latency(_metric(row, "mean_inference_ms")),
        ]
        lines.append(" & ".join(values) + r" \\")

    caption = (
        f"{scenario_title} lane-following: observation space ablation "
        f"(dense reward, no obstacles, 30~eval episodes)."
    )
    return _table_lines(
        "table",
        caption,
        table_label,
        "lcccccc",
        [
            r"Observation"
            r"  & \makecell{CTE\\Tractor\\(m)}"
            r"  & \makecell{CTE\\Trailer\\(m)}"
            r"  & \makecell{Max $|\gamma|$\\($^\circ$)}"
            r"  & \makecell{Completion\\(\%)}"
            r"  & \makecell{Jackknife\\(\%)}"
            r"  & \makecell{Inference\\(ms)} \\",
        ],
        lines,
    )


def _generate_reward_ablation_table(
    aggregates: dict[str, dict[str, str]],
    scenario: str,
    best_obs_tag: str,
) -> str:
    scenario_title = "Forward" if scenario == "forward" else "Reverse"
    rewards = (
        [("Dense", "dense"), ("Tractor focus", "tractor_focus"), ("Multiplicative", "multiplicative"), ("Guided", "guided")]
        if scenario == "forward"
        else [("Dense", "dense"), ("No hitch", "no_hitch"), ("Multiplicative", "multiplicative"), ("Guided", "guided")]
    )
    table_label = "tab:reward_ablation_forward" if scenario == "forward" else "tab:reward_ablation_reverse"

    body_rows: list[str] = []
    for label_text, reward in rewards:
        row = _eval_row(aggregates, f"{scenario}/{best_obs_tag}/{reward}")
        values = [
            label_text,
            _fmt_float(_metric(row, "mean_abs_cte_tractor")),
            _fmt_float(_metric(row, "mean_abs_cte_trailer")),
            _fmt_degrees(_metric(row, "max_abs_hitch_angle")),
            _fmt_percent(_metric(row, "completed")),
            _fmt_percent(_metric(row, "jackknifed")),
            _fmt_float(_metric(row, "total_reward"), digits=2),
        ]
        body_rows.append(" & ".join(values) + r" \\")

    caption = (
        f"{scenario_title} lane-following: reward formulation ablation "
        f"({_humanize_obs_tag(best_obs_tag)} obs, no obstacles, 30~eval episodes)."
    )
    return _table_lines(
        "table",
        caption,
        table_label,
        "lcccccc",
        [
            r"Reward"
            r"  & \makecell{CTE\\Tractor\\(m)}"
            r"  & \makecell{CTE\\Trailer\\(m)}"
            r"  & \makecell{Max $|\gamma|$\\($^\circ$)}"
            r"  & \makecell{Completion\\(\%)}"
            r"  & \makecell{Jackknife\\(\%)}"
            r"  & \makecell{Total\\reward} \\",
        ],
        body_rows,
    )


def _generate_lidar_table(aggregates: dict[str, dict[str, str]]) -> str:
    beam_tags = [("4", "lidar_4"), ("8", "lidar_8"), ("16", "lidar"), ("24", "lidar_24"), ("32", "lidar_32")]
    body_rows: list[str] = []
    for beam_label, obs_tag in beam_tags:
        row = _eval_row(aggregates, f"forward/{obs_tag}/dense")
        values = [
            beam_label,
            _fmt_float(_metric(row, "mean_abs_cte_trailer")),
            _fmt_float(_metric(row, "rms_cte_trailer")),
            _fmt_degrees(_metric(row, "max_abs_hitch_angle")),
            _fmt_percent(_metric(row, "completed")),
            _fmt_percent(_metric(row, "jackknifed")),
            _fmt_latency(_metric(row, "mean_inference_ms")),
        ]
        body_rows.append(" & ".join(values) + r" \\")

    caption = (
        "Lidar beam-count ablation on forward lane-following "
        "(dense reward, no obstacles, 30~eval episodes)."
    )
    return _table_lines(
        "table",
        caption,
        "tab:lidar_beams_forward",
        "rcccccc",
        [
            r"Beams"
            r"  & \makecell{CTE\\Trailer\\mean (m)}"
            r"  & \makecell{CTE\\Trailer\\RMS (m)}"
            r"  & \makecell{Max $|\gamma|$\\($^\circ$)}"
            r"  & \makecell{Completion\\(\%)}"
            r"  & \makecell{Jackknife\\(\%)}"
            r"  & \makecell{Inference\\(ms)} \\",
        ],
        body_rows,
    )


def _generate_failure_replay_table(aggregates: dict[str, dict[str, str]], cfg: Config) -> str:
    labels = [
        ("Random reset", f"reverse/{cfg.best_obs_reverse}/{cfg.best_reward_reverse}"),
        ("Failure replay", f"reverse/{cfg.best_obs_reverse}/{cfg.best_reward_reverse}_retry"),
    ]
    body_rows: list[str] = []
    for title, label in labels:
        row = _eval_row(aggregates, label)
        values = [
            title,
            "---",
            _fmt_float(_metric(row, "mean_abs_cte_trailer")),
            _fmt_degrees(_metric(row, "max_abs_hitch_angle")),
            _fmt_percent(_metric(row, "completed")),
            _fmt_percent(_metric(row, "jackknifed")),
        ]
        body_rows.append(" & ".join(values) + r" \\")

    caption = (
        "Failure replay curriculum ablation on reverse lane-following "
        f"({_humanize_obs_tag(cfg.best_obs_reverse)} obs, {_humanize_reward(cfg.best_reward_reverse)} reward, "
        "30~held-out eval episodes)."
    )
    return _table_lines(
        "table",
        caption,
        "tab:failure_replay",
        "lccccc",
        [
            r"Reset strategy"
            r"  & \makecell{Steps to\\50\% completion\\($\times 10^3$)}"
            r"  & \makecell{CTE\\Trailer\\(m)}"
            r"  & \makecell{Max $|\gamma|$\\($^\circ$)}"
            r"  & \makecell{Completion\\(\%)}"
            r"  & \makecell{Jackknife\\(\%)} \\",
        ],
        body_rows,
    )


def _generate_obstacle_table(aggregates: dict[str, dict[str, str]], scenario: str, cfg: Config) -> str:
    scenario_title = "Forward" if scenario == "forward_obs" else "Reverse"
    reward = cfg.best_reward_forward if scenario == "forward_obs" else cfg.best_reward_reverse
    table_label = "tab:obstacle_forward" if scenario == "forward_obs" else "tab:obstacle_reverse"

    rows = [
        ("State (MLP)", f"{scenario}/state/{reward}"),
        (f"{_lidar_display_name(cfg.obstacle_lidar_tag)} (MLP)", f"{scenario}/{cfg.obstacle_lidar_tag}/{reward}"),
        ("BEV CNN (scratch)", f"{scenario}/bev/{reward}"),
    ]

    body_rows: list[str] = []
    for title, label in rows:
        row = _eval_row(aggregates, label)
        values = [
            title,
            _fmt_float(_metric(row, "mean_abs_cte_trailer")),
            _fmt_degrees(_metric(row, "max_abs_hitch_angle")),
            _fmt_percent(_metric(row, "completed")),
            _fmt_percent(_metric(row, "collided")),
            _fmt_percent(_metric(row, "jackknifed")),
            _fmt_float(_metric(row, "min_clearance_m")),
            _fmt_latency(_metric(row, "mean_inference_ms")),
        ]
        body_rows.append(" & ".join(values) + r" \\")

    caption = (
        f"{scenario_title} obstacle-avoidance: TD3 performance by observation space "
        f"({_humanize_reward(reward)} reward, 5--10 obstacles per episode, 30~eval episodes)."
    )
    return _table_lines(
        "table",
        caption,
        table_label,
        "lccccccc",
        [
            r"Observation"
            r"  & \makecell{CTE\\Trailer\\(m)}"
            r"  & \makecell{Max $|\gamma|$\\($^\circ$)}"
            r"  & \makecell{Path\\completion\\(\%)}"
            r"  & \makecell{Collision\\(\%)}"
            r"  & \makecell{Jackknife\\(\%)}"
            r"  & \makecell{Min clearance\\(m)}"
            r"  & \makecell{Inference\\(ms)} \\",
        ],
        body_rows,
    )


def _aggregate_controller_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        ctrl = row.get("controller", "")
        grouped.setdefault(ctrl, []).append(row)

    metrics = [
        "mean_abs_cte_trailer",
        "rms_cte_trailer",
        "max_abs_hitch_angle",
        "completed",
        "jackknifed",
        "mean_inference_ms",
    ]
    aggregated: dict[str, dict[str, float | None]] = {}
    for ctrl, ctrl_rows in grouped.items():
        aggregated[ctrl] = {}
        for metric in metrics:
            parser = _parse_boolish if metric in {"completed", "jackknifed"} else _parse_float
            vals = [parser(row.get(metric)) for row in ctrl_rows]
            vals = [val for val in vals if val is not None]
            aggregated[ctrl][metric] = (sum(vals) / len(vals)) if vals else None
    return aggregated


def _generate_benchmark_table(rows: list[dict[str, str]], task: str) -> str:
    scenario_title = "Forward" if task == "forward" else "Reverse"
    table_label = "tab:benchmark_forward" if task == "forward" else "tab:benchmark_reverse"
    controller_order = (
        [("TD3 (best config)", "td3"), ("Forward pure pursuit", "fpp"), ("PID (tuned)", "pid"), ("MPC (tuned)", "mpc")]
        if task == "forward"
        else [("TD3 (best config)", "td3"), ("Reverse pure pursuit", "fpp_rev"), ("PID reverse (tuned)", "pid_rev"), ("MPC reverse (tuned)", "mpc_rev")]
    )
    aggregated = _aggregate_controller_rows(rows)

    body_rows: list[str] = []
    for title, ctrl in controller_order:
        row = aggregated.get(ctrl)
        values = [
            title,
            _fmt_float(row.get("mean_abs_cte_trailer") if row else None),
            _fmt_float(row.get("rms_cte_trailer") if row else None),
            _fmt_degrees(row.get("max_abs_hitch_angle") if row else None),
            _fmt_percent(row.get("completed") if row else None),
            _fmt_percent(row.get("jackknifed") if row else None),
            _fmt_latency(row.get("mean_inference_ms") if row else None),
        ]
        body_rows.append(" & ".join(values) + r" \\")

    caption = (
        f"{scenario_title} lane-following: controller benchmark "
        "(no obstacles, 50~eval episodes, held-out seeds)."
    )
    return _table_lines(
        "table",
        caption,
        table_label,
        "lcccccc",
        [
            r"Controller"
            r"  & \makecell{CTE\\Trailer\\mean (m)}"
            r"  & \makecell{CTE\\Trailer\\RMS (m)}"
            r"  & \makecell{Max $|\gamma|$\\($^\circ$)}"
            r"  & \makecell{Completion\\(\%)}"
            r"  & \makecell{Jackknife\\(\%)}"
            r"  & \makecell{Latency\\(ms)} \\",
        ],
        body_rows,
    )


def _parse_run_script_defaults(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pattern = re.compile(r'^([A-Z_]+)="?([^"\n]+)"?$')
    defaults: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            defaults[match.group(1)] = match.group(2)
    return defaults


def _build_config(args: argparse.Namespace) -> Config:
    defaults = _parse_run_script_defaults(args.run_script)
    lidar_beams = str(args.lidar_beams_default or defaults.get("LIDAR_BEAMS_DEFAULT", "16"))
    obs_lidar_tag = args.obs_lidar_tag or (f"lidar_{lidar_beams}" if lidar_beams != "16" else "lidar")
    obstacle_lidar_tag = args.obstacle_lidar_tag or obs_lidar_tag

    return Config(
        results_dir=args.results_dir,
        tables_dir=args.tables_dir,
        obs_lidar_tag=obs_lidar_tag,
        obstacle_lidar_tag=obstacle_lidar_tag,
        best_obs_forward=args.best_obs_forward or defaults.get("BEST_OBS_FORWARD", "state"),
        best_obs_reverse=args.best_obs_reverse or defaults.get("BEST_OBS_REVERSE", "state"),
        best_reward_forward=args.best_reward_forward or defaults.get("BEST_REWARD_FORWARD", "dense"),
        best_reward_reverse=args.best_reward_reverse or defaults.get("BEST_REWARD_REVERSE", "dense"),
    )


def generate_tables(cfg: Config) -> dict[str, str]:
    aggregates = _load_eval_aggregates(cfg.results_dir)
    benchmark_forward_rows = _load_benchmark_rows(cfg.results_dir, "forward")
    benchmark_reverse_rows = _load_benchmark_rows(cfg.results_dir, "reverse")

    return {
        "obs_ablation_results.tex": (
            "% Auto-generated by thesis/scripts/generate_results_tables.py\n\n"
            + _generate_obs_ablation_table(aggregates, "forward", cfg)
            + _generate_obs_ablation_table(aggregates, "reverse", cfg)
        ),
        "reward_ablation_results.tex": (
            "% Auto-generated by thesis/scripts/generate_results_tables.py\n\n"
            + _generate_reward_ablation_table(aggregates, "forward", cfg.best_obs_forward)
            + _generate_reward_ablation_table(aggregates, "reverse", cfg.best_obs_reverse)
        ),
        "lidar_beams_ablation.tex": (
            "% Auto-generated by thesis/scripts/generate_results_tables.py\n\n"
            + _generate_lidar_table(aggregates)
        ),
        "failure_replay_results.tex": (
            "% Auto-generated by thesis/scripts/generate_results_tables.py\n\n"
            + _generate_failure_replay_table(aggregates, cfg)
        ),
        "obstacle_results.tex": (
            "% Auto-generated by thesis/scripts/generate_results_tables.py\n\n"
            + _generate_obstacle_table(aggregates, "forward_obs", cfg)
            + _generate_obstacle_table(aggregates, "reverse_obs", cfg)
        ),
        "benchmark_results.tex": (
            "% Auto-generated by thesis/scripts/generate_results_tables.py\n\n"
            + _generate_benchmark_table(benchmark_forward_rows, "forward")
            + _generate_benchmark_table(benchmark_reverse_rows, "reverse")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis LaTeX tables from results CSVs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tables-dir", type=Path, default=Path("thesis/tables"))
    parser.add_argument("--run-script", type=Path, default=Path("run_ablations.sh"))
    parser.add_argument("--lidar-beams-default", type=int, default=None)
    parser.add_argument("--obs-lidar-tag", default=None, help="Obs-ablation lidar tag, e.g. lidar or lidar_24")
    parser.add_argument("--obstacle-lidar-tag", default=None, help="Obstacle-table lidar tag, e.g. lidar or lidar_24")
    parser.add_argument("--best-obs-forward", default=None)
    parser.add_argument("--best-obs-reverse", default=None)
    parser.add_argument("--best-reward-forward", default=None)
    parser.add_argument("--best-reward-reverse", default=None)
    args = parser.parse_args()

    cfg = _build_config(args)
    outputs = generate_tables(cfg)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in outputs.items():
        path = cfg.tables_dir / filename
        path.write_text(content)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
