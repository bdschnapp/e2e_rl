"""Evaluate/tune the classical controllers (PP, PID) in the batched env.

Handles stateful controllers (PID integral) by passing the done-mask so the
integrator resets on episode boundaries. Used to (a) verify feasibility, (b) pick
the best classical controller (the guided reward's teacher), and (c) provide the
classical baselines for the RL-vs-classical benchmark.
"""
import sys, os, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from tractor_trailer_rl.batched import backend
backend.set_backend("numpy")
from run_ablations_gpu import cell_to_cfg
from tractor_trailer_rl.batched.sb3_adapter import SB3BatchedVecEnv
from tractor_trailer_rl.batched.pure_pursuit import BatchedPurePursuit
from tractor_trailer_rl.batched import pid as pidmod
from tractor_trailer_rl.batched.lqr import BatchedLQR
from tractor_trailer_rl.batched.mpc_tractor import KinematicTractorMPC

JACK = 1.4


def eval_controller(ctrl, env, steps=800):
    o = env.reset()
    if hasattr(ctrl, "reset"):
        ctrl.reset(env.num_envs)
    N = env.num_envs
    ep_cte = np.zeros(N); ep_len = np.zeros(N); ep_maxh = np.zeros(N)
    comps, ctes, hitches = [], [], []
    dones = None
    for _ in range(steps):
        ep_maxh = np.maximum(ep_maxh, np.abs(o[:, 1])); ep_cte += np.abs(o[:, 4]); ep_len += 1
        a, _ = ctrl.predict(o, dones=dones) if "dones" in ctrl.predict.__code__.co_varnames else ctrl.predict(o)
        o, r, dones, infos = env.step(a)
        for i in np.nonzero(dones)[0]:
            comps.append(float(infos[i].get("is_success", 0.0)))
            ctes.append(ep_cte[i] / max(ep_len[i], 1)); hitches.append(ep_maxh[i])
            ep_cte[i] = ep_len[i] = ep_maxh[i] = 0.0
    m = lambda x: float(np.mean(x)) if x else float("nan")
    return dict(completion=m(comps), trailer_cte=m(ctes), max_hitch=m(hitches), n=len(comps))


def tune_pid(direction, cfg, env, n_iter=120, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(n_iter):
        if direction == "reverse":
            g = dict(k_hitch=rng.uniform(-3, 3), Kp=rng.uniform(0, 2), Ki=rng.uniform(0, 0.05),
                     Kd=rng.uniform(0, 3), k_ff=rng.uniform(-3, 3), max_integral=0.30)
        else:
            g = dict(Kp=rng.uniform(0, 1), Ki=rng.uniform(0, 0.1), Kd=rng.uniform(0, 3),
                     Kp_t=rng.uniform(0, 2), k_ff=rng.uniform(-2, 2), max_integral=0.30)
        m = eval_controller(pidmod.BatchedPID(cfg, reverse=(direction == "reverse"), gains=g), env, 600)
        cte = m["trailer_cte"] if m["trailer_cte"] == m["trailer_cte"] else 1e9
        key = (round(m["completion"], 3), -cte)
        if best is None or key > best[0]:
            best = (key, g, m)
    return best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_file", default=None,
                    help="evaluate on this PP-validated safe pool (same feasible paths as the "
                         "RL eval) for an apples-to-apples RL-vs-classical comparison")
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    cargs = ap.parse_args()
    # Classical baseline benchmark: the four controllers spanning the control spectrum
    # (geometric -> classical feedback -> optimal linear -> optimal constrained+preview),
    # evaluated in the same batched env as the RL policies for a fair comparison.
    for direction in ["forward", "reverse"]:
        cfg = cell_to_cfg(f"{direction}:state:multiplicative", preset="shunt",
                          fixed_speed=True, mild_paths=True, pool_file=cargs.pool_file)
        env = SB3BatchedVecEnv(cfg, cargs.n_envs, path_pool_size=128)
        rev = direction == "reverse"
        ctrls = {"PurePursuit": BatchedPurePursuit(cfg, reverse=rev),
                 "PID": pidmod.BatchedPID(cfg, reverse=rev),
                 "LQR": BatchedLQR(cfg, rev),
                 "MPC": KinematicTractorMPC(cfg, rev)}
        print(f"[{direction}]")
        for name, ctrl in ctrls.items():
            m = eval_controller(ctrl, env, cargs.steps)
            print(f"    {name:12s} compl={m['completion']:.3f} cte={m['trailer_cte']:.3f} "
                  f"hitch={m['max_hitch']:.3f}")
