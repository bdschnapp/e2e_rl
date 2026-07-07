"""Prune continuous-algo (ppo/td3/sac) runs from a Stage-1 sweep dir so they can be
re-run on the corrected 2-D STOP_SIGNAL action space, keeping the already-valid
discrete cells (dqn/ppo_disc, which are 2-D Discrete(n*2)).

The lane-following ablation had continuous cells trained on a 1-D FIXED_SPEED action
(no stop) — an inconsistency vs the 2-D stop-capable requirement. This removes those
rows from manifest.jsonl + curves.csv (backing both up to *.prebak) so a follow-up
    run_stage1_sweep.py --algos ppo td3 --outdir <dir>
re-fills them on 2-D and recomputes summary.csv over the full, symmetric 96-cell set.
"""
import sys, os, json, shutil, csv

CONT = {"ppo", "td3", "sac"}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results_stage1_v3"
    man = os.path.join(d, "manifest.jsonl")
    cur = os.path.join(d, "curves.csv")
    shutil.copy(man, man + ".prebak")
    shutil.copy(cur, cur + ".prebak")

    def is_continuous(run_id, algo):
        if algo in CONT:
            return True
        # error records have a run_id but no algo field
        return run_id and run_id.split("__")[0] in CONT

    keep, dropped = [], 0
    for line in open(man):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            keep.append(line); continue
        if is_continuous(rec.get("run_id", ""), rec.get("algo", "")):
            dropped += 1
        else:
            keep.append(line)
    with open(man, "w") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))

    rows = list(csv.DictReader(open(cur)))
    fields = rows[0].keys() if rows else ["run_id", "algo"]
    kept = [r for r in rows if r.get("algo") not in CONT]
    with open(cur, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(kept)

    print(f"pruned {dropped} continuous manifest runs; kept {len(keep)} manifest rows")
    print(f"curves.csv rows {len(rows)} -> {len(kept)}  (backups: *.prebak)")


if __name__ == "__main__":
    main()
