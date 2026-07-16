"""Phase-2B vision rescue via BC warm-start + SDF image (NEW file; additive, remote-run).

Convergent finding (Ryan's projects + lit + our failed screen): a from-scratch CNN
co-trained with the TD3 critic destabilises a policy the state vector already solves.
The fix is to DECOUPLE the encoder from RL reward. Cheapest high-evidence route:
behavior-clone the working state expert into an SDF-BEV policy (the CNN + actor learn
under a supervised loss first), THEN TD3 fine-tune.

Expert = the 2A state-forward TD3 policy (solves the lane task at 1.0). We roll it out
inside the SDF-BEV env (its action depends only on the state vector, which is identical
in both obs), recording (Dict obs, expert action), and clone the vision actor on them.

Touches nothing run_stage1_sweep depends on. Run on the remote (numpy/torch/cupy venv2).

    TTRL_BACKEND=cupy python3 -u scripts/bc_rescue.py \
        --expert results_stage2_obs/models/td3__forward__state__multiplicative__s0.zip \
        --coord --bc_steps 600 --bc_epochs 8 --steps 400000 --seed 0
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
from train_sb3 import evaluate, EvalCurveCallback  # noqa: E402
from sb3_stability import SdfVecEnv, build_td3, stability_extractor_class  # noqa: E402
from tractor_trailer_rl.batched.bev_sdf_env import SdfObsCfg  # noqa: E402

POOL = "path_pools/shunt_mild_safe.npz"


def make_cfgs():
    """forward / multiplicative / shunt / mild — matches the 2A state cell the expert
    was trained on (beams=0 => the BEV 'vector' is the same 8-dim state the expert sees)."""
    cfg = cell_to_cfg("forward:bev:multiplicative", preset="shunt",
                      fixed_speed=True, mild_paths=True)
    cfg = replace(cfg, action=replace(cfg.action, mode=ActionMode.STOP_SIGNAL))
    eval_cfg = replace(cfg, path=replace(cfg.path, pool_file=POOL)) if os.path.exists(POOL) else cfg
    return cfg, eval_cfg


def collect(expert, env, n_steps):
    """Roll out the state EXPERT (driven by obs['vector']) in the SDF-BEV env; record the
    Dict obs it saw + the action it took. Returns numpy arrays (vector, image, action)."""
    obs = env.reset()
    V, I, A = [], [], []
    for _ in range(n_steps):
        a, _ = expert.predict(obs["vector"], deterministic=True)   # state MlpPolicy
        V.append(np.asarray(obs["vector"], dtype=np.float32).copy())
        I.append(np.asarray(obs["image"], dtype=np.float32).copy())
        A.append(np.asarray(a, dtype=np.float32).copy())
        obs, _, _, _ = env.step(a)
    return np.concatenate(V), np.concatenate(I), np.concatenate(A)


def bc_train(model, data, epochs, batch):
    """Supervised warm-start of the TD3 actor (incl. its CNN encoder) to imitate the
    expert. SB3 actor outputs are in [-1,1]; scale expert actions to match."""
    import torch as th
    V, I, A = data
    n = len(A)
    A_scaled = model.policy.scale_action(A)                 # -> [-1,1]
    A_t = th.as_tensor(A_scaled, dtype=th.float32, device=model.device)
    model.policy.set_training_mode(True)
    opt = model.actor.optimizer
    for ep in range(epochs):
        idx = np.random.permutation(n)
        tot = 0.0; nb = 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            obs_np = {"vector": V[b], "image": I[b]}
            t_obs, _ = model.policy.obs_to_tensor(obs_np)
            pred = model.actor(t_obs)                        # [-1,1] action
            loss = ((pred - A_t[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        print(f"  BC epoch {ep+1}/{epochs}: mse={tot/max(nb,1):.5f}", flush=True)
    model.actor_target.load_state_dict(model.actor.state_dict())


def _unfreeze_head_callback(model, warmup_steps, unfreeze_lr=1e-3):
    """At warmup_steps, unfreeze the actor HEAD only (encoder stays frozen) and rebuild
    the actor optimizer over just those params at `unfreeze_lr` (a lower LR gives a gentler,
    dip-free adaptation against the warmed critic), so RL adapts control without collapse."""
    from stable_baselines3.common.callbacks import BaseCallback
    import torch as th

    class _UF(BaseCallback):
        def __init__(self):
            super().__init__(); self.done = False

        def _on_step(self):
            if not self.done and self.num_timesteps >= warmup_steps:
                head = []
                for name, p in model.actor.named_parameters():
                    if "features_extractor" not in name:   # head only; encoder stays frozen
                        p.requires_grad = True; head.append(p)
                if head:
                    model.actor.optimizer = th.optim.Adam(head, lr=unfreeze_lr)
                self.done = True
                print(f"# UNFROZE actor head at step {self.num_timesteps} "
                      f"({len(head)} tensors, lr={unfreeze_lr}; encoder still frozen)", flush=True)
            return True

    return _UF()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="results_stage2_obs/models/td3__forward__state__multiplicative__s0.zip")
    ap.add_argument("--coord", action="store_true", help="add 2 CoordConv channels to the SDF image")
    ap.add_argument("--bc_steps", type=int, default=600, help="env steps of expert rollout for BC data (xN_envs transitions)")
    ap.add_argument("--bc_collect_envs", type=int, default=256)
    ap.add_argument("--bc_epochs", type=int, default=8)
    ap.add_argument("--bc_batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400_000, help="TD3 fine-tune budget")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_stability", action="store_true", help="use the plain BEV extractor instead of the LayerNorm/sep one")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="after BC, freeze the CNN feature extractors (actor+critic) so RL "
                         "fine-tuning cannot corrupt the imitation-learned representation")
    ap.add_argument("--freeze_actor", action="store_true",
                    help="after BC, freeze the ENTIRE actor (encoder+head); RL only learns the "
                         "critic (policy stays exactly the BC policy)")
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="critic warm-up schedule: keep the actor frozen for the first N steps "
                         "(critic trains against the good BC policy), THEN unfreeze the actor HEAD "
                         "(encoder stays frozen) so RL adapts control against a warmed critic. Phase-3 recipe.")
    ap.add_argument("--unfreeze_lr", type=float, default=1e-3,
                    help="actor LR after unfreeze (lower => gentler, dip-free adaptation)")
    args = ap.parse_args()

    from stable_baselines3 import TD3
    print(f"# BC+SDF rescue backend={backend.get_backend()} coord={args.coord} "
          f"bc_steps={args.bc_steps} bc_epochs={args.bc_epochs} steps={args.steps} seed={args.seed}", flush=True)

    sdf_cfg = SdfObsCfg(args.coord)
    cfg, eval_cfg = make_cfgs()

    # target TD3 model over the SDF-BEV env (LayerNorm + separate actor/critic encoders)
    train_env = SdfVecEnv(cfg, args.n_envs, sdf_cfg=sdf_cfg, path_pool_size=1024)
    ext = None if args.no_stability else stability_extractor_class()
    share = None if args.no_stability else False
    model = build_td3(train_env, args.seed, extractor_class=ext, share_features_extractor=share)

    # eval env (safe pool, no-stop) — the same protocol as run_stage1_sweep
    eval_env = SdfVecEnv(eval_cfg, 256, sdf_cfg=sdf_cfg, path_pool_size=512)
    eval_env.env._eval_no_stop = True

    if not os.path.exists(args.expert):
        raise FileNotFoundError(f"expert model not found: {args.expert}")
    expert = TD3.load(args.expert, device="cuda")
    print(f"# loaded expert {args.expert}", flush=True)

    # 1) collect expert transitions in the SDF-BEV env
    t0 = time.perf_counter()
    collect_env = SdfVecEnv(cfg, args.bc_collect_envs, sdf_cfg=sdf_cfg, path_pool_size=1024)
    data = collect(expert, collect_env, args.bc_steps)
    print(f"# collected {len(data[2])} transitions in {time.perf_counter()-t0:.0f}s "
          f"(img shape {data[1].shape[1:]})", flush=True)

    # 2) BC warm-start the actor (+ its encoder)
    bc_train(model, data, args.bc_epochs, args.bc_batch)
    bc_eval = evaluate(model, eval_env, steps=800)
    print(f"# BC-ONLY (no RL yet): completion={bc_eval['completion']:.3f} "
          f"cte={bc_eval['trailer_cte']:.3f} hitch={bc_eval['max_hitch']:.3f}", flush=True)

    # 2b) optionally protect the imitation-learned policy from RL corruption
    if args.freeze_encoder or args.freeze_actor:
        import torch as th
        frozen = 0
        for net in (model.actor, model.critic):
            for name, p in net.named_parameters():
                keep_frozen = (args.freeze_actor and net is model.actor) or \
                              ("features_extractor" in name)
                if keep_frozen:
                    p.requires_grad = False; frozen += 1
            trainable = [p for p in net.parameters() if p.requires_grad]
            net.optimizer = th.optim.Adam(trainable, lr=1e-3) if trainable else net.optimizer
        tag = "actor(full)+encoders" if args.freeze_actor else "encoders(actor+critic)"
        print(f"# froze {tag}: {frozen} tensors; RL trains the rest only", flush=True)

    # 3) TD3 fine-tune from the BC init
    cb = EvalCurveCallback(eval_env, args.eval_every)
    callbacks = cb
    if args.warmup_steps > 0:
        import torch as th
        # freeze BOTH encoders for the WHOLE run; freeze the actor head too during warm-up
        for net in (model.actor, model.critic):
            for name, p in net.named_parameters():
                if "features_extractor" in name or net is model.actor:
                    p.requires_grad = False
            tr = [p for p in net.parameters() if p.requires_grad]
            if tr:
                net.optimizer = th.optim.Adam(tr, lr=1e-3)   # critic head trains during warm-up
        print(f"# encoders FROZEN for the whole run; actor frozen for a {args.warmup_steps}-step "
              f"critic warm-up, then the actor HEAD is unfrozen (encoder stays frozen)", flush=True)
        callbacks = [cb, _unfreeze_head_callback(model, args.warmup_steps, args.unfreeze_lr)]
    t1 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=callbacks)
    best = cb.best or bc_eval
    fin = evaluate(model, eval_env, steps=800)
    # best-CTE among fully-completing checkpoints => "did RL improve tracking past the BC/expert ceiling?"
    solved = [c for c in cb.curve if c.get("completion", 0) >= 0.99]
    best_cte_solved = min((c["trailer_cte"] for c in solved), default=float("nan"))
    print(f"# BC+RL DONE ({time.perf_counter()-t1:.0f}s): "
          f"BEST completion={best['completion']:.3f} cte={best['trailer_cte']:.3f} @{best.get('timesteps')} "
          f"| final completion={fin['completion']:.3f} cte={fin['trailer_cte']:.3f} "
          f"| best_cte@completion>=0.99={best_cte_solved:.3f} (BC-only cte was {bc_eval['trailer_cte']:.3f})", flush=True)
    print("# curve (t, compl, cte):",
          [(c['timesteps'], round(c['completion'], 3), round(c['trailer_cte'], 3)) for c in cb.curve], flush=True)
    import csv as _csv
    _out = f"bc_curve_s{args.seed}_warm{args.warmup_steps}_lr{args.unfreeze_lr}.csv"
    with open(_out, "w", newline="") as _f:
        _flds = ["timesteps", "completion", "trailer_cte", "max_hitch", "jackknife", "ep_return", "n_episodes"]
        _w = _csv.DictWriter(_f, fieldnames=_flds); _w.writeheader()
        for c in cb.curve:
            _w.writerow({k: c.get(k) for k in _flds})
    print(f"# wrote curve csv -> {_out}", flush=True)


if __name__ == "__main__":
    main()
