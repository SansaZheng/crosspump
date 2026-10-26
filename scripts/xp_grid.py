#!/usr/bin/env python3
"""Parallel CrossPump grid on one server: datasets x methods x seeds, each job a subprocess of scripts/xp_run.py,
run at nice 10 with one BLAS/torch thread. Finished jobs (summary.json present) are skipped, so the grid is resumable.

    .venv/bin/python scripts/xp_grid.py --workers 40 --datasets all --methods all --seeds 42 2024 2025 2026
    .venv/bin/python scripts/xp_grid.py --workers 8 --datasets hu_pd --methods xp_counts_rule --seeds 42 --max-folds 1 --tag probe
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["hu_pd", "sapienza_pd", "xu_pumpolymp", "ardia_pd", "pumpsense", "tgdataset_pd", "mirtaheri", "dex_wash", "rugpull1k", "memechain"]
METHODS = ["crosspump", "xp_counts_rule", "xp_mirtaheri21", "xp_nghiem21", "xp_victor21", "xp_lamorgia23", "xp_hu23", "xp_bolz24", "xp_perseus25",
           "xp_pumpsense26", "xp_moura26", "xp_lroo26"]
ABLATIONS = ["counts_only", "hawkes_only", "llr_only", "single_model", "no_social", "no_wallet", "no_orderflow", "social_only", "orderflow_only", "fixed_gate"]


def ablation_override(name):
    import yaml, json
    d = yaml.safe_load(open(os.path.join(ROOT, "configs", "crosspump", "ablations", f"{name}.yaml")))
    d.pop("_base_", None); d.pop("ablation", None)
    return [f"baseline.{k}={json.dumps(v)}" for k, v in d.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--datasets", nargs="*", default=["all"]); ap.add_argument("--methods", nargs="*", default=["all"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 2024, 2025, 2026]); ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None); ap.add_argument("--tag", default=None); ap.add_argument("--device", default="cpu"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ds = DATASETS if a.datasets == ["all"] else a.datasets
    ms = METHODS if a.methods == ["all"] else a.methods
    jobs = []
    for d in ds:
        for s in a.seeds:
            for m in ms:
                jobs.append((d, m, s, a.tag, []))
            if a.ablations:
                for ab in ABLATIONS:
                    jobs.append((d, "crosspump", s, f"abl_{ab}", ablation_override(ab)))
    logdir = os.path.join(ROOT, "results", "crosspump", "logs"); os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONWARNINGS="ignore")

    def run(job):
        d, m, s, tag, over = job
        done = os.path.join(ROOT, "results", "crosspump", d, m, tag or "main", f"seed{s}", "summary.json")
        if os.path.exists(done):
            return job, "skip", 0.0
        cmd = ["nice", "-n", "10", sys.executable, os.path.join(ROOT, "scripts", "xp_run.py"), "--dataset", d, "--baseline", m, "--seed", str(s), "--device", a.device]
        if tag: cmd += ["--tag", tag]
        if a.max_folds is not None: cmd += ["--max-folds", str(a.max_folds)]
        if a.smoke: cmd += ["--smoke"]
        if over: cmd += ["--override"] + over
        t0 = time.time()
        with open(os.path.join(logdir, f"{d}_{m}_{tag or 'main'}_s{s}.log"), "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        return job, ("ok" if rc == 0 else f"rc={rc}"), time.time() - t0

    # build each dataset's panel once first (serially per dataset, in parallel across datasets) so jobs hit the cache
    def warm(d):
        r = subprocess.call(["nice", "-n", "10", sys.executable, "-c", f"import sys; sys.path.insert(0,'scripts'); from xp_common import *; import argparse; import_all(); "
                             f"a=argparse.Namespace(dataset='{d}', baseline='crosspump', smoke={a.smoke}, override=[], max_folds=None); c,dc,_=load_cfgs(a); b,p=build_panel(dc); "
                             f"print('{d}', p.T, p.N, p.meta.get('n_pumps'), p.meta.get('coverage'))"],
                            stdout=open(os.path.join(logdir, f"panel_{d}.log"), "w"), stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        return d, r
    with ThreadPoolExecutor(min(len(ds), a.workers)) as ex:
        ok_ds = [d for d, r in ex.map(warm, ds) if r == 0 or print(f"[panel FAILED] {d}: see results/crosspump/logs/panel_{d}.log", flush=True)]
    print(f"[panels ok] {ok_ds}", flush=True)
    jobs = [j for j in jobs if j[0] in ok_ds]
    print(f"[grid] {len(jobs)} jobs on {a.workers} workers", flush=True)
    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            job, st, dt = fut.result(); n += 1
            print(f"[{n}/{len(jobs)}] {st:6s} {dt:6.0f}s {job[0]} {job[1]} {job[3] or 'main'} seed{job[2]}", flush=True)


if __name__ == "__main__":
    main()
