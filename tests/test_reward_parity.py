"""Stage 2 gate: the reward composer must reproduce e2e_rl's per-step running
reward for ALL supported reward modes (dense, tractor_focus, multiplicative for
forward; dense, no_hitch, multiplicative for reverse), within 1e-5 — incl. the
fix B additive floor and fix C proximity threshold.

Fixtures (dump_oracle_fixtures.py) record reward_<mode> at each snapshot.
"""
import os
from dataclasses import replace
import numpy as np

from tractor_trailer_rl import (
    build_observation, EgoState, TrailerState, lab_config,
    Direction, VehicleKind, GridMeta,
)
from tractor_trailer_rl.reward.composer import running_reward
from tractor_trailer_rl.reward.proximity import proximity_penalty

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CASES = {
    "forward_trailer": (Direction.FORWARD, VehicleKind.TRAILER),
    "reverse_trailer": (Direction.REVERSE, VehicleKind.TRAILER),
    "forward_tractor_only": (Direction.FORWARD, VehicleKind.TRACTOR_ONLY),
    "reverse_tractor_only": (Direction.REVERSE, VehicleKind.TRACTOR_ONLY),
}
ATOL = 1e-5


def _check(name):
    d = np.load(os.path.join(FIX, f"{name}.npz"))
    direction, kind = CASES[name]
    base = lab_config(direction, kind)
    occ = d["occ"]
    meta = GridMeta(origin_x=float(d["occ_origin"][0]), origin_y=float(d["occ_origin"][1]),
                    res_m=float(d["occ_res"]), width=occ.shape[1], height=occ.shape[0])
    xx, yy = d["xx"], d["yy"]
    modes = [k[len("reward_"):] for k in d.files if k.startswith("reward_")]
    results = {}
    n = len(d["vx"])
    for mode in modes:
        cfg = replace(base, reward=replace(base.reward, mode=mode))
        rew = d[f"reward_{mode}"]
        max_err = 0.0
        for i in range(n):
            if abs(float(rew[i])) > 90.0:
                continue  # terminal step (+/-100/200/500) — running reward N/A
            ego = EgoState(x=float(d["vx"][i]), y=float(d["vy"][i]), yaw=float(d["vp"][i]),
                           steer=float(d["vs"][i]), xd=float(d["vxd"][i]))
            tr = TrailerState(x=float(d["tx"][i]), y=float(d["ty"][i]), yaw=float(d["tyaw"][i]))
            e = build_observation(xx, yy, ego, tr, cfg, occ_grid=occ, occ_meta=meta).errors
            prox = proximity_penalty(occ, meta, [(ego.x, ego.y), (tr.x, tr.y)], cfg)
            got = running_reward(
                cfg, e_y=e["e_y"], e_psi=e["e_psi"],
                e_y_t=e.get("e_y_t", 0.0), e_psi_t=e.get("e_psi_t", 0.0),
                hitch=e.get("hitch", 0.0), xd=ego.xd, proximity=prox)
            max_err = max(max_err, abs(got - float(rew[i])))
        results[mode] = max_err
        assert max_err <= ATOL, f"{name}/{mode}: reward max err {max_err:.2e} > {ATOL}"
    return results


def test_forward_trailer(): _check("forward_trailer")
def test_reverse_trailer(): _check("reverse_trailer")
def test_forward_tractor_only(): _check("forward_tractor_only")
def test_reverse_tractor_only(): _check("reverse_tractor_only")


if __name__ == "__main__":
    for nm in CASES:
        try:
            res = _check(nm)
            print(f"PASS {nm:22s} " + " ".join(f"{m}={e:.1e}" for m, e in res.items()))
        except AssertionError as ex:
            print(f"FAIL {ex}")
