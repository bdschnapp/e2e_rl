"""Phase-2B geometry-encoder pretraining (NEW file, additive; remote-run).

Supervised distillation: train a CNN to REGRESS the physical state from the SDF-BEV
image ALONE (privileged teacher = simulator ground-truth). The immediate deliverable is
PER-COMPONENT RECOVERABILITY (R^2 per state dim), which tells us exactly which parts of
the state the ego-centric image contains (expected: tractor e_y / e_psi / curvature yes;
steering delta / hitch gamma no). That split then defines the RL observation.

The frozen encoder is saved for the RL stage (RL on estimated geometry, from scratch).

    TTRL_BACKEND=cupy python3 -u scripts/geom_pretrain.py --coord \
        --expert results_stage2_obs/models/td3__forward__state__multiplicative__s0.zip \
        --steps 800 --epochs 20 --out geom_encoder_fwd_coord.pt
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
from dataclasses import replace  # noqa: E402
from tractor_trailer_rl.batched import backend  # noqa: E402
from tractor_trailer_rl.config import ActionMode  # noqa: E402
from run_ablations_gpu import cell_to_cfg  # noqa: E402
from sb3_stability import SdfVecEnv  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402
from geom_encoder import GeomEncoder  # noqa: E402

# trailer state layout (beams=0): [steer, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2]
NAMES8 = ["delta(steer)", "gamma(hitch)", "e_y", "e_psi", "e_y_t", "e_psi_t", "k1", "k2"]


def make_cfg(direction):
    cfg = cell_to_cfg(f"{direction}:bev:multiplicative", preset="shunt",
                      fixed_speed=True, mild_paths=True)
    return replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))


def collect(expert, env, n_steps, rand_frac, act_dim, rng):
    """Roll out the expert (on-task distribution) with a fraction of random actions
    (coverage). Record (image, true_state)."""
    obs = env.reset()
    n = env.num_envs
    IMG, ST = [], []
    for t in range(n_steps):
        a_exp, _ = expert.predict(obs["vector"], deterministic=True)
        a = a_exp.copy()
        if rand_frac > 0:
            m = rng.random(n) < rand_frac
            a[m] = rng.uniform(-1.0, 1.0, size=(int(m.sum()), act_dim)).astype(np.float32)
        IMG.append(np.asarray(obs["image"], dtype=np.float32).copy())
        ST.append(np.asarray(obs["vector"], dtype=np.float32).copy())
        obs, _, _, _ = env.step(a)
    return np.concatenate(IMG), np.concatenate(ST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--expert", default="results_stage2_obs/models/td3__forward__state__multiplicative__s0.zip")
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=800, help="env steps of rollout (xN_envs samples)")
    ap.add_argument("--rand_frac", type=float, default=0.25, help="fraction of random actions for coverage")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="geom_encoder.pt")
    args = ap.parse_args()

    import torch as th
    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)
    print(f"# geom-pretrain backend={backend.get_backend()} dir={args.direction} coord={args.coord} "
          f"steps={args.steps} epochs={args.epochs}", flush=True)

    from stable_baselines3 import TD3
    sdf_cfg = SdfObsCfg(args.coord)
    cfg = make_cfg(args.direction)
    env = SdfVecEnv(cfg, args.n_envs, sdf_cfg=sdf_cfg, path_pool_size=1024)
    act_dim = int(env.action_space.shape[0])
    expert = TD3.load(args.expert, device="cuda")

    t0 = time.perf_counter()
    IMG, ST = collect(expert, env, args.steps, args.rand_frac, act_dim, rng)
    in_ch = IMG.shape[1]; out_dim = ST.shape[1]; N = len(ST)
    print(f"# collected {N} samples (img {IMG.shape[1:]}, state dim {out_dim}) in {time.perf_counter()-t0:.0f}s", flush=True)

    # standardize labels per-dim (so R^2 = 1 - MSE_standardized, scale-invariant)
    mu = ST.mean(0); sd = ST.std(0) + 1e-6
    STn = (ST - mu) / sd
    # 90/10 split
    idx = rng.permutation(N); ntr = int(0.9 * N)
    tr, te = idx[:ntr], idx[ntr:]
    dev = "cuda"
    Xtr = th.as_tensor(IMG[tr], device=dev); Ytr = th.as_tensor(STn[tr], dtype=th.float32, device=dev)
    Xte = th.as_tensor(IMG[te], device=dev); Yte = th.as_tensor(STn[te], dtype=th.float32, device=dev)

    enc = GeomEncoder(in_ch, out_dim, img_size=IMG.shape[-1]).to(dev)
    opt = th.optim.Adam(enc.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        enc.train(); perm = th.randperm(ntr, device=dev); tot = 0.0; nb = 0
        for i in range(0, ntr, args.batch):
            b = perm[i:i + args.batch]
            pred = enc(Xtr[b]); loss = ((pred - Ytr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            enc.eval()
            with th.no_grad():
                r2 = 1.0 - ((enc(Xte) - Yte) ** 2).mean(0).cpu().numpy()  # per-dim (std labels)
            print(f"  ep {ep+1}/{args.epochs} train_mse={tot/nb:.4f} test_R2_mean={float(np.mean(r2)):.3f}", flush=True)

    enc.eval()
    with th.no_grad():
        r2 = 1.0 - ((enc(Xte) - Yte) ** 2).mean(0).cpu().numpy()
    names = NAMES8 if out_dim == 8 else [f"dim{i}" for i in range(out_dim)]
    print("# PER-COMPONENT RECOVERABILITY (test R^2; ~1 = image contains it, ~0 = it does not):", flush=True)
    for nm, v in zip(names, r2):
        print(f"    {nm:14s} R2={v:+.3f}", flush=True)

    th.save({"state_dict": enc.state_dict(), "in_ch": in_ch, "out_dim": out_dim,
             "img_size": int(IMG.shape[-1]), "label_mean": mu, "label_std": sd,
             "names": names, "direction": args.direction, "coord": args.coord}, args.out)
    print(f"# saved encoder -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
