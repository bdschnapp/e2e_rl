"""Warm-start corner fine-tune: continue each mild baseline on the FULL path
mixture (corners in), keeping obs (lidar+state) + the clean stop handling.
Usage: MODELS="forward_trailer ..." STEPS=200000 python scripts/finetune_corners.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER","dummy"); os.environ.setdefault("SDL_AUDIODRIVER","dummy")
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"1")
os.environ.setdefault("PYTHONUNBUFFERED","1")
from dataclasses import replace
import numpy as np
from tractor_trailer_rl import lab_config, Direction, VehicleKind, ActionMode
from tractor_trailer_rl.train.runner import finetune

MODELS={"forward_trailer":(Direction.FORWARD,VehicleKind.TRAILER),
        "reverse_trailer":(Direction.REVERSE,VehicleKind.TRAILER),
        "forward_tractor_only":(Direction.FORWARD,VehicleKind.TRACTOR_ONLY),
        "reverse_tractor_only":(Direction.REVERSE,VehicleKind.TRACTOR_ONLY)}

def cfg_for(direction,kind):
    c=lab_config(direction,kind)
    return replace(c,
        action=replace(c.action, mode=ActionMode.STOP_SIGNAL, stop_threshold=0.8,
                       stop_noise_sigma=0.05, stop_penalty=500.0),
        reward=replace(c.reward, mode="multiplicative"),
        speed_random=replace(c.speed_random, explicit_min=0.5, explicit_max=0.8),
        path=replace(c.path, mild_only=False))

def main():
    steps=int(os.environ.get("STEPS",200000)); n_envs=int(os.environ.get("N_ENVS",8))
    base=os.environ.get("BASE", os.path.expanduser("~/Ben/Thesis/Electrans_project/lab_models_ttrl_baseline"))
    outroot=os.environ.get("OUT", os.path.expanduser("~/Ben/Thesis/Electrans_project/lab_models_ttrl_finetuned"))
    want=os.environ.get("MODELS"," ".join(MODELS)).split()
    for name in want:
        direction,kind=MODELS[name]
        init=os.path.join(base,name,"best_model.zip")
        out=os.path.join(outroot,name)
        print(f"\n===== FINETUNE {name}  from {init}  steps={steps} =====",flush=True)
        finetune(init, cfg_for(direction,kind), timesteps=steps, n_envs=n_envs,
                 out_dir=out, eval_freq=20000, n_eval_episodes=10, device="auto")
        d=np.load(os.path.join(out,"logs","evaluations.npz")); res=d["results"].mean(1); ln=d["ep_lengths"].mean(1)
        b=int(np.argmax(res))
        print(f"  {name}: best={res[b]:.1f}/len{ln[b]:.0f}@{int(d['timesteps'][b])//1000}k final={res[-1]:.1f}/len{ln[-1]:.0f}",flush=True)
    print("\nFINETUNE DONE",flush=True)

if __name__=="__main__":
    main()
