"""
Generate the hardware-deployment results table and figures for Chapter 6.

Reads the CSV extracts of the A2B deployment rosbags recorded on 2026-08-05
(thesis/data/deployment_bags/csv/, produced by extract_deployment_bags.py on
the robot) and writes:

  thesis/tables/deployment_runs.tex
  thesis/figures/experiment_results/deployment_trajectories.pdf
  thesis/figures/experiment_results/deployment_tracking.pdf
  thesis/data/deployment_bags/deployment_summary.csv

Nine bags were recorded that day. RUNS below selects the four that completed
the maneuver under policy control, two forward and two reverse; run
survey_all_runs() to re-screen every bag and see why the other five are
excluded (one stationary bench session, four reverse operator takeovers).

Metric definitions
------------------
Evaluation window ("policy-controlled segment"): ticks where the lane-reference
node reports drive_enabled in the trial's direction AND the bridge is
commanding a non-zero velocity in that direction. The window closes at the
first zero-velocity command sustained for >= 1 s; everything after it
(including any manual recovery drive) is excluded from the statistics.

Ground truth is geometric and independent of the policy's own observation. The
tractor reference point is /localization/kinematic_state (Autoware base_link =
rear-axle centre), which is also the hitch location. The trailer axle is
reconstructed from the onboard hitch estimate gamma as

    p_t = p_tractor - L_t * [cos(psi - gamma), sin(psi - gamma)],   L_t = 2.8 m

Cross-track error is the signed perpendicular offset to the reference
centerline polyline, positive to the left of the direction of travel. Heading
error is the body heading relative to the centerline tangent taken along the
direction of travel, plus pi (the body faces backwards while reversing).

Usage
-----
    python thesis/scripts/generate_deployment_results.py
"""

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THESIS = HERE.parent
DATA = THESIS / "data" / "deployment_bags"
CSVROOT = DATA / "csv"
TABLES = THESIS / "tables"
FIGS = THESIS / "figures" / "experiment_results"

# ---------------------------------------------------------------- constants
L_TRAILER = 2.8        # hitch -> trailer axle, m (rl_bridge_node.py)
LANE_HALF_W = 1.41     # lane corridor half-width, m (rl_bridge_node.py)
HITCH_OP_LIMIT = 80.0  # deployment operational articulation limit, deg
JACKKNIFE = 90.0       # simulation jackknife termination, deg
STOP_HOLD_S = 1.0      # sustained zero-velocity that closes the window

# Validated categorical palette (dataviz six-checks, light surface: all PASS).
# Secondary encoding: solid = forward, dashed = reverse.
COLORS = ["#2A5DB0", "#00916E", "#D55E00", "#A03070"]

# The four successful A-to-B trials: two forward, two reverse. All nine bags
# recorded on 2026-08-05 were screened (see survey_all_runs()); the remaining
# five are one stationary bench session and four reverse trials ended by
# operator takeover.
RUNS = [
    ("run_A2B_fwd_20260805_100756", "F1", "10:07:56"),
    ("run_A2B_fwd_20260805_101802", "F2", "10:18:02"),
    ("run_A2B_rev_20260805_120951", "R1", "12:09:51"),
    ("run_A2B_rev_20260805_124549", "R2", "12:45:49"),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def polyline_frame(path, pts):
    """Signed lateral offset, arclength and tangent heading w.r.t. a polyline."""
    seg_a, seg_b = path[:-1], path[1:]
    d = seg_b - seg_a
    L2 = (d ** 2).sum(axis=1)
    L2[L2 == 0] = 1e-12
    seglen = np.sqrt(L2)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    P, A, D = pts[:, None, :], seg_a[None, :, :], d[None, :, :]
    w = np.clip(((P - A) * D).sum(axis=2) / L2[None, :], 0.0, 1.0)
    proj = A + w[:, :, None] * D
    dist = np.linalg.norm(P - proj, axis=2)
    j = np.argmin(dist, axis=1)
    n = np.arange(len(pts))
    dv = d[j]
    tang = np.arctan2(dv[:, 1], dv[:, 0])
    rel = pts - proj[n, j]
    cross = dv[:, 0] * rel[:, 1] - dv[:, 1] * rel[:, 0]
    return np.sign(cross) * dist[n, j], cum[j] + w[n, j] * seglen[j], tang


def oscillation_stats(hitch_deg, s_path):
    """Characterize the articulation limit cycle: spatial wavelength between
    successive up-crossings of zero, and mean peak-to-peak amplitude."""
    h = hitch_deg - hitch_deg.mean()
    up = np.where((h[:-1] <= 0) & (h[1:] > 0))[0]
    if len(up) < 3:
        return dict(osc_wavelength_m=float("nan"), osc_amp_pp_deg=float("nan"),
                    osc_cycles=len(up))
    lam = np.abs(np.diff(s_path[up]))
    amps = []
    for i in range(len(up) - 1):
        seg = hitch_deg[up[i]:up[i + 1]]
        if len(seg) > 2:
            amps.append(seg.max() - seg.min())
    return dict(
        osc_wavelength_m=round(float(np.median(lam)), 2),
        osc_amp_pp_deg=round(float(np.median(amps)), 1) if amps else float("nan"),
        osc_cycles=int(len(up) - 1),
    )


def analyze(run):
    d = CSVROOT / run
    rd = lambda f: pd.read_csv(d / f)
    ks = rd("localization__kinematic_state.csv")
    tr = rd("vehicle__trailer_state.csv")
    st = rd("vehicle__status__steering_status.csv")
    cmd = rd("control__command__control_cmd.csv")
    en = rd("planning__lane_reference__drive_enabled.csv")
    dr = rd("planning__lane_reference__drive_direction.csv")
    sv = rd("rl_bridge__state_vector.csv")
    idx = rd("centerline_index.csv")
    cl = np.load(d / "centerlines.npz")

    t = sv.t_ns.values.astype(np.float64)          # one stamp per policy tick
    x = np.interp(t, ks.t_ns.values, ks.x.values)
    y = np.interp(t, ks.t_ns.values, ks.y.values)
    yaw = wrap(np.interp(t, ks.t_ns.values, np.unwrap(ks.yaw.values)))
    vx = np.interp(t, ks.t_ns.values, ks.vx.values)
    hitch = np.interp(t, tr.t_ns.values, tr.hitch_angle.values)
    steer = np.interp(t, st.t_ns.values, st.steering_tire_angle.values)
    srate = np.interp(t, cmd.t_ns.values, cmd.steer_rate_cmd.values)
    vcmd = np.interp(t, cmd.t_ns.values, cmd.vel_cmd.values)

    def hold(df, col):
        j = np.clip(np.searchsorted(df.t_ns.values, t, side="right") - 1,
                    0, len(df) - 1)
        return df[col].values[j]

    enabled = hold(en, "enabled").astype(bool)
    reverse = hold(dr, "reverse").astype(bool)
    pid = int(pd.Series(hold(idx, "path_id")[enabled]).mode().iloc[0])
    path = cl[f"p{pid}"]

    # Direction of the trial, from the lane-reference node's own flag.
    is_rev = bool(pd.Series(reverse[enabled]).mode().iloc[0])
    # "Moving" = commanding motion in the intended direction.
    moving = (vcmd < -1e-6) if is_rev else (vcmd > 1e-6)

    tractor = np.stack([x, y], 1)
    yaw_t = wrap(yaw - hitch)
    trailer = np.stack([x - L_TRAILER * np.cos(yaw_t),
                        y - L_TRAILER * np.sin(yaw_t)], 1)

    lat, s_path, tang = polyline_frame(path, tractor)
    lat_t, s_path_t, tang_t = polyline_frame(path, trailer)

    driving = enabled & (reverse == is_rev) & moving
    travel_sign = -1.0 if np.median(np.gradient(s_path)[driving]) < 0 else 1.0
    off = np.pi if travel_sign < 0 else 0.0
    e_y = lat * travel_sign
    e_y_t = lat_t * travel_sign
    # Reversing, the body faces opposite the direction of travel; going
    # forward it faces along it.
    body = np.pi if is_rev else 0.0
    e_psi = wrap(yaw - (tang + off + body))
    e_psi_t = wrap(yaw_t - (tang_t + off + body))

    dt = float(np.median(np.diff(t)) / 1e9)
    hold_n = max(1, int(round(STOP_HOLD_S / dt)))
    stopped = ~moving
    start = int(np.where(driving)[0][0])
    end = len(t)
    for i in range(start, len(t) - hold_n):
        if stopped[i:i + hold_n].all():
            end = i
            break
    win = np.zeros(len(t), bool)
    win[start:end] = True
    win &= enabled & (reverse == is_rev)
    a = win

    # Takeover signature: after the window closes the vehicle keeps moving
    # against the commanded (zero) velocity, i.e. under manual control.
    post = np.zeros(len(t), bool)
    post[end:] = True
    counter = (vx > 0.05) if is_rev else (vx < -0.05)
    takeover = bool((post & counter & ~moving).sum() > 10)
    progress = float(abs(s_path[a][-1] - s_path[a][0]))
    hd = np.degrees(hitch)

    res = dict(
        run=run,
        direction="reverse" if is_rev else "forward",
        bag_duration_s=round(float((t[-1] - t[0]) / 1e9), 1),
        window_s=round(float(a.sum() * dt), 1),
        path_progress_m=round(progress, 2),
        dist_driven_m=round(float(np.hypot(np.diff(x[a]), np.diff(y[a])).sum()), 2),
        mean_speed_mps=round(float(np.abs(vx[a]).mean()), 3),
        completed=bool(progress > 20.0 and not takeover),
        operator_takeover=takeover,
        cte_mean_abs_m=round(float(np.abs(e_y[a]).mean()), 3),
        cte_rms_m=round(float(np.sqrt((e_y[a] ** 2).mean())), 3),
        cte_max_abs_m=round(float(np.abs(e_y[a]).max()), 3),
        cte_t_mean_abs_m=round(float(np.abs(e_y_t[a]).mean()), 3),
        cte_t_rms_m=round(float(np.sqrt((e_y_t[a] ** 2).mean())), 3),
        cte_t_max_abs_m=round(float(np.abs(e_y_t[a]).max()), 3),
        frac_trailer_outside_lane=round(float((np.abs(e_y_t[a]) > LANE_HALF_W).mean()), 3),
        epsi_mean_abs_deg=round(float(np.degrees(np.abs(e_psi[a])).mean()), 2),
        epsi_max_abs_deg=round(float(np.degrees(np.abs(e_psi[a])).max()), 2),
        epsi_t_mean_abs_deg=round(float(np.degrees(np.abs(e_psi_t[a])).mean()), 2),
        epsi_t_max_abs_deg=round(float(np.degrees(np.abs(e_psi_t[a])).max()), 2),
        hitch_mean_abs_deg=round(float(np.abs(hd[a]).mean()), 2),
        hitch_max_abs_deg=round(float(np.abs(hd[a]).max()), 2),
        frac_hitch_over_op_limit=round(float((np.abs(hd[a]) > HITCH_OP_LIMIT).mean()), 4),
        jackknife=bool((np.abs(hd[a]) >= JACKKNIFE).any()),
        steer_max_abs_deg=round(float(np.degrees(np.abs(steer[a])).max()), 2),
        steer_rate_sat_frac=round(float((np.abs(srate[a]) > 0.99 * 0.436332).mean()), 3),
    )
    res.update(oscillation_stats(hd[a], s_path[a]))

    series = pd.DataFrame(dict(
        t_s=(t - t[0]) / 1e9, x=x, y=y, yaw=yaw, vx=vx, hitch_deg=hd,
        steer_rad=steer, steer_rate_cmd=srate, vel_cmd=vcmd,
        trailer_x=trailer[:, 0], trailer_y=trailer[:, 1],
        e_y=e_y, e_y_t=e_y_t, e_psi_deg=np.degrees(e_psi),
        e_psi_t_deg=np.degrees(e_psi_t), s_path=s_path,
        enabled=enabled, reverse=reverse, in_window=win,
    ))
    series["progress_m"] = np.abs(series.s_path.values - s_path[win][0])
    return res, series, path


# ------------------------------------------------------------------ outputs
def write_table(df):
    lab = {r[0]: (r[1], r[2]) for r in RUNS}
    def row(r):
        tag, clock = lab[r["run"]]
        return (f"{tag} ({clock}) & {r['path_progress_m']:.1f} & "
                f"{r['mean_speed_mps']:.2f} & "
                f"{r['cte_mean_abs_m']:.3f} & {r['cte_max_abs_m']:.2f} & "
                f"{r['cte_t_mean_abs_m']:.3f} & {r['cte_t_max_abs_m']:.2f} & "
                f"{r['hitch_mean_abs_deg']:.1f} & {r['hitch_max_abs_deg']:.1f} & "
                f"{100 * r['steer_rate_sat_frac']:.0f} \\\\")

    lines = [
        "% Auto-generated by thesis/scripts/generate_deployment_results.py",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Lane following on the physical platform: the four successful "
        r"A-to-B trials recorded on 2026-08-05, two in each direction, over the "
        r"same surveyed lane segment. Statistics cover the policy-controlled "
        r"segment of each trial and are computed geometrically from the "
        r"localization estimate against the reference centerline, independently "
        r"of the policy's own observation. The final column is the fraction of "
        r"control ticks at which the commanded steering rate sat on its "
        r"saturation limit.}",
        r"\label{tab:deployment_runs}",
        r"\resizebox{\ifdim\width>\linewidth \linewidth\else \width\fi}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Trial & \makecell{Driven\\(m)} & \makecell{Speed\\(m/s)} "
        r"& \multicolumn{2}{c}{Tractor CTE (m)} "
        r"& \multicolumn{2}{c}{Trailer CTE (m)} "
        r"& \multicolumn{2}{c}{$|\gamma|$ (deg)} & \makecell{$\dot{\delta}$ sat\\(\%)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r" & & & mean & max & mean & max & mean & max & \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\textbf{Forward}} \\",
    ]
    fwd = df[df.direction == "forward"]
    rev = df[df.direction == "reverse"]
    lines += [row(r) for _, r in fwd.iterrows()]
    lines += [r"\addlinespace", r"\multicolumn{10}{l}{\textbf{Reverse}} \\"]
    lines += [row(r) for _, r in rev.iterrows()]
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    (TABLES / "deployment_runs.tex").write_text("\n".join(lines))
    print(f"wrote {TABLES / 'deployment_runs.tex'}")


def plot_trajectories(results, series, paths):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    allx = np.concatenate([series[r][series[r].in_window].x.values
                           for r, _, _ in RUNS])
    ally = np.concatenate([series[r][series[r].in_window].y.values
                           for r, _, _ in RUNS])
    x0, x1 = allx.min() - 3.5, allx.max() + 3.5
    y0, y1 = ally.min() - 3.0, ally.max() + 3.0

    # reference lane, clipped to the driven region
    path = paths[RUNS[0][0]]
    keep = ((path[:, 0] > x0 - 2) & (path[:, 0] < x1 + 2) &
            (path[:, 1] > y0 - 2) & (path[:, 1] < y1 + 2))
    ax.plot(path[keep, 0], path[keep, 1], color="0.55", lw=1.6, ls=(0, (5, 3)),
            zorder=1, label="Reference centerline")

    for (run, tag, _), c in zip(RUNS, COLORS):
        w = series[run][series[run].in_window]
        rev = results[run]["direction"] == "reverse"
        ls = (0, (4, 2)) if rev else "-"
        ax.plot(w.x, w.y, color=c, lw=1.7, ls=ls, zorder=3,
                label=f"{tag} ({results[run]['direction']})")
        ax.plot(w.x.iloc[-1], w.y.iloc[-1], marker="o", ms=6,
                color=c, mec="white", mew=1.2, zorder=4, ls="none")

    w0 = series[RUNS[0][0]][series[RUNS[0][0]].in_window]
    ax.plot(w0.x.iloc[0], w0.y.iloc[0], marker="s", ms=8,
            color="0.2", mec="white", mew=1.2, zorder=5, ls="none")
    ax.annotate("A (start)", (w0.x.iloc[0], w0.y.iloc[0]),
                textcoords="offset points", xytext=(0, 11), fontsize=8, color="0.25")
    wb = series[RUNS[0][0]][series[RUNS[0][0]].in_window]
    ax.annotate("B (goal)", (wb.x.iloc[-1], wb.y.iloc[-1]),
                textcoords="offset points", xytext=(-44, -6), fontsize=8, color="0.25")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("map $x$ (m)")
    ax.set_ylabel("map $y$ (m)")
    ax.legend(loc="lower left", frameon=False, ncol=2, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"deployment_trajectories.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'deployment_trajectories.pdf'}")


def plot_tracking(results, series):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.4, 5.7), sharex=True)

    # (a) trailer cross-track error
    ax1.axhspan(-LANE_HALF_W, LANE_HALF_W, color="0.88", zorder=0)
    ax1.axhline(0, color="0.6", lw=0.7, zorder=1)
    for (run, tag, _), c in zip(RUNS, COLORS):
        w = series[run][series[run].in_window]
        ls = (0, (4, 2)) if results[run]["direction"] == "reverse" else "-"
        ax1.plot(w.progress_m, w.e_y_t, color=c, lw=1.4, ls=ls, label=tag, zorder=3)
    ax1.set_ylabel("Trailer CTE (m)")
    ax1.annotate("lane corridor", (0.3, -LANE_HALF_W + 0.12), fontsize=7,
                 color="0.45", va="bottom")
    ax1.legend(loc="upper left", frameon=False, ncol=4, columnspacing=1.4)
    ax1.set_title("(a) Trailer cross-track error", loc="left", color="0.3")

    # (b) articulation angle
    for (run, _, _), c in zip(RUNS, COLORS):
        w = series[run][series[run].in_window]
        ls = (0, (4, 2)) if results[run]["direction"] == "reverse" else "-"
        ax2.plot(w.progress_m, w.hitch_deg, color=c, lw=1.4, ls=ls, zorder=3)
    for lim, lab_ in ((HITCH_OP_LIMIT, "operational limit"), (JACKKNIFE, "jackknife")):
        ax2.axhline(lim, color="0.45", lw=0.8, ls=":")
        ax2.axhline(-lim, color="0.45", lw=0.8, ls=":")
    ax2.annotate("jackknife", (0.3, JACKKNIFE), fontsize=7, color="0.45", va="bottom")
    ax2.annotate("operational limit", (0.3, -HITCH_OP_LIMIT), fontsize=7,
                 color="0.45", va="top")
    ax2.set_ylim(-118, 112)
    ax2.set_ylabel(r"$\gamma$ (deg)")
    ax2.set_title("(b) Articulation angle", loc="left", color="0.3")

    # (c) commanded steering rate
    sat = 0.436332
    for (run, _, _), c in zip(RUNS, COLORS):
        w = series[run][series[run].in_window]
        ls = (0, (4, 2)) if results[run]["direction"] == "reverse" else "-"
        ax3.plot(w.progress_m, w.steer_rate_cmd, color=c, lw=1.1, ls=ls, zorder=3)
    ax3.axhline(sat, color="0.45", lw=0.8, ls=":")
    ax3.axhline(-sat, color="0.45", lw=0.8, ls=":")
    ax3.annotate("rate limit", (0.3, sat), fontsize=7, color="0.45", va="bottom")
    ax3.set_ylim(-0.62, 0.68)
    ax3.set_ylabel(r"$\dot{\delta}_{\mathrm{cmd}}$ (rad/s)")
    ax3.set_xlabel("Distance traveled along reference path (m)")
    ax3.set_title("(c) Commanded steering rate", loc="left", color="0.3")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"deployment_tracking.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'deployment_tracking.pdf'}")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    results, series, paths, rows = {}, {}, {}, []
    for run, _, _ in RUNS:
        res, ser, path = analyze(run)
        results[run], series[run], paths[run] = res, ser, path
        rows.append(res)
        ser.to_csv(DATA / f"{run}_series.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "deployment_summary.csv", index=False)
    (DATA / "deployment_summary.json").write_text(json.dumps(rows, indent=2))

    write_table(df)
    plot_trajectories(results, series, paths)
    plot_tracking(results, series)

    comp = df[df.completed]
    print("\n--- summary ---")
    print(f"completed {len(comp)}/{len(df)}; "
          f"completed-run trailer CTE mean {comp.cte_t_mean_abs_m.mean():.2f} m, "
          f"max {comp.cte_t_max_abs_m.max():.2f} m; "
          f"peak |gamma| {df.hitch_max_abs_deg.max():.1f} deg; "
          f"jackknife {df.jackknife.any()}")


def survey_all_runs():
    """Screen every bag under csv/ and print why each is kept or excluded.

    Used to choose RUNS. Run with:  python <this file> --survey
    """
    rows = []
    for run in sorted(os.listdir(CSVROOT)):
        try:
            res, _, _ = analyze(run)
        except Exception as e:
            rows.append(dict(run=run, direction="?", note=f"no drive window ({e})"))
            continue
        res["note"] = ("completed" if res["completed"]
                       else ("operator takeover" if res["operator_takeover"]
                             else "incomplete"))
        rows.append(res)
    df = pd.DataFrame(rows)
    df["run"] = df.run.str.replace("run_A2B_", "", regex=False)
    cols = [c for c in ["run", "direction", "note", "path_progress_m",
                        "cte_mean_abs_m", "cte_t_mean_abs_m", "cte_t_max_abs_m",
                        "hitch_max_abs_deg", "steer_rate_sat_frac"]
            if c in df.columns]
    pd.set_option("display.width", 220)
    print(df[cols].to_string(index=False))
    print(f"\nselected for the thesis: {[t for _, t, _ in RUNS]}")


if __name__ == "__main__":
    import sys

    if "--survey" in sys.argv:
        survey_all_runs()
    else:
        main()
