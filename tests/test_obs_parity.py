"""Stage 1 gate: tractor_trailer_rl.build_observation must reproduce the e2e_rl
oracle obs within 1e-5 for all 4 concrete lidar_24 envs (fwd/rev x trailer/
tractor-only). Fixtures from dump_oracle_fixtures.py.
"""
import os
import numpy as np

from tractor_trailer_rl import (
    build_observation, EgoState, TrailerState, lab_config,
    Direction, VehicleKind, GridMeta,
)

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
    cfg = lab_config(direction, kind)
    occ = d["occ"]
    meta = GridMeta(origin_x=float(d["occ_origin"][0]), origin_y=float(d["occ_origin"][1]),
                    res_m=float(d["occ_res"]), width=occ.shape[1], height=occ.shape[0])
    xx, yy = d["xx"], d["yy"]
    obs_oracle = d["obs"]
    n = len(obs_oracle)
    max_err = 0.0
    worst = None
    for i in range(n):
        ego = EgoState(x=float(d["vx"][i]), y=float(d["vy"][i]), yaw=float(d["vp"][i]),
                       steer=float(d["vs"][i]), xd=float(d["vxd"][i]))
        tr = TrailerState(x=float(d["tx"][i]), y=float(d["ty"][i]), yaw=float(d["tyaw"][i]))
        got = build_observation(xx, yy, ego, tr, cfg, occ_grid=occ, occ_meta=meta).vector
        err = np.abs(got - obs_oracle[i])
        if err.max() > max_err:
            max_err = float(err.max()); worst = (i, int(err.argmax()))
    assert max_err <= ATOL, (
        f"{name}: max obs error {max_err:.2e} > {ATOL} at (step,dim)={worst}")
    return max_err


def test_forward_trailer():
    print("forward_trailer max_err", _check("forward_trailer"))


def test_reverse_trailer():
    print("reverse_trailer max_err", _check("reverse_trailer"))


def test_forward_tractor_only():
    print("forward_tractor_only max_err", _check("forward_tractor_only"))


def test_reverse_tractor_only():
    print("reverse_tractor_only max_err", _check("reverse_tractor_only"))


if __name__ == "__main__":
    for nm in CASES:
        try:
            e = _check(nm)
            print(f"PASS {nm:22s} max_err={e:.2e}")
        except AssertionError as ex:
            print(f"FAIL {ex}")
