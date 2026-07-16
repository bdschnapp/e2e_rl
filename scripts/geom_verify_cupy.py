"""Verify the CuPy encoder forward matches the torch encoder (Phase-2B; NEW file).
Runs both on the same batch of BEV images and reports the prediction discrepancy.

    TTRL_BACKEND=cupy python3 -u scripts/geom_verify_cupy.py --encoder geom_encoder_fwd_coord.pt --coord
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
from dataclasses import replace  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.batched.backend import xp, to_numpy  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from sb3_stability import SdfVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402
from tractor_trailer_rl.batched.geom_encoder_cupy import GeomEncoderCupy  # noqa: E402
from geom_encoder import GeomEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    import torch as th
    ckpt = th.load(args.encoder, weights_only=False)

    te = GeomEncoder(ckpt["in_ch"], ckpt["out_dim"], ckpt["img_size"]).to("cuda")
    te.load_state_dict(ckpt["state_dict"]); te.eval()
    tmean = th.as_tensor(ckpt["label_mean"], dtype=th.float32, device="cuda")
    tstd = th.as_tensor(ckpt["label_std"], dtype=th.float32, device="cuda")
    ce = GeomEncoderCupy(ckpt)

    cfg = cell_to_cfg("forward:bev:multiplicative", preset="shunt", fixed_speed=True, mild_paths=True)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    env = SdfVecEnv(cfg, args.n, sdf_cfg=SdfObsCfg(args.coord), path_pool_size=512)
    img = np.asarray(env.reset()["image"], dtype=np.float32)

    with th.no_grad():
        tp = (te(th.as_tensor(img, device="cuda")) * tstd + tmean).cpu().numpy()
    cp_pred = to_numpy(ce.predict(xp.asarray(img)))

    d = np.abs(tp - cp_pred)
    print(f"# backend={backend.get_backend()} img {img.shape} torch-vs-cupy prediction diff:", flush=True)
    print(f"    max_abs={d.max():.2e}  mean_abs={d.mean():.2e}  (should be ~1e-4 or less)", flush=True)
    print("    per-dim max_abs:", np.array2string(d.max(0), precision=2, floatmode="maxprec"), flush=True)
    print("MATCH" if d.max() < 1e-2 else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
