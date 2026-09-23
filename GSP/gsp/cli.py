"""Command line: `gsp <group> <command>` or `python -m gsp.cli ...`.

S1:  gsp instances freeze [--N 4 5 ...] [--draws 30] [--results DIR] [--replace-set]
     gsp instances verify [--deep] [--results DIR]
     gsp instances table  [--no-write] [--results DIR]
     gsp index            [--results DIR]     (rebuild results/index/registry.parquet)
S2:  gsp sectors build    [--N ...] [--no-extension] [--force] [--results DIR]
     gsp sectors tables | report [--no-write]
     gsp sectors pybind   --module-dir DIR --label NAME [--rules ...] [--N ...] [--seed-study N_SEEDS]
S3:  gsp mixer counts                 (gate counts (ii)/(iii)/T of every confined cell and instance)
     gsp mixer c1 [--engine numpy|cudaq] [--decomposed-n-max 10]   (C1 operator level, real sectors n <= 12)
     gsp mixer time                   (GPU timing of the A1 circuit; one GPU process)
     gsp mixer report [--no-write]    (reports/mixer_counts.md)
S4:  gsp arms smoke                   (every §1.2 cell at N = 4: A0 / A1 / A2p / A2c runs, validated; GPU)
     gsp arms anneal                  (A2 anneal check, both encodings, both ramp signs; GPU)
     gsp arms repro                   (A1 / A0 reproduction vs the completed runs' AR2; GPU)
     gsp arms time                    (s / iteration before / after the S4 speedups; GPU, sequential subprocesses)
     gsp arms report [--no-write]     (reports/arms.md from the tables above and s4_legacy.json)
     gsp arms run --arm A1 --inst N04e004q1.5 --effort 5 [--K 12 --rule violation --conn ring] [--lam ...]
     (scripts/s4_legacy_checks.py: the legacy-equivalence numbers, results/tables/s4_legacy.json)
S5:  gsp metrics finalize [--arms A0 ...] [--limit N] [--force]
                                      (GPU: samples.npz + postrun.json for the stored runs that lack them)
     gsp metrics checks               (the S5 evidence -> results/tables/s5_checks.json)
     gsp metrics report [--no-write]  (reports/metrics.md)
     gsp aggregate [--rebuild]        (results/tables/metrics.parquet, one row per done run; incremental)
     gsp stats gbar                   (results/tables/gbar.parquet: gbar_arm(N) of Rule D1 from the frozen set)
     gsp stats d1 [--lam-star 4=0.005 5=...] [--ring-order lex]
                                      (Rule D1 over the stored runs -> results/tables/d1.parquet, d1_choices.parquet)
Later sessions add plan, run, report, missing, selfcheck.
"""

from __future__ import annotations

from . import _threads  # noqa: F401  (pins BLAS/OpenMP threads before numpy loads)

import argparse
import json
import sys


def _cmd_instances(args) -> int:
    from .instances import freeze as fz
    from .instances.draws import DRAWS_PER_N, N_VALUES

    if args.action == "freeze":
        Ns = tuple(args.N) if args.N else N_VALUES
        fz.freeze(root=args.results, dataset_dir=args.dataset, N_values=Ns,
                  draws_per_n=args.draws or DRAWS_PER_N, replace_set=args.replace_set)
        ok = fz.verify(root=args.results, dataset_dir=args.dataset)
        return 0 if ok else 1
    if args.action == "verify":
        ok = fz.verify(root=args.results, deep=args.deep, dataset_dir=args.dataset)
        return 0 if ok else 1
    if args.action == "table":
        text = fz.table_markdown(args.results)
        print(text)
        if not args.no_write:
            p = fz.write_report(args.results)
            print(f"\nwritten to {p}", file=sys.stderr)
        return 0
    raise AssertionError(args.action)


def _cmd_index(args) -> int:
    from .store.index import build_index

    df = build_index(args.results)
    print(f"indexed {len(df)} run(s)")
    if not df.empty and "status" in df:
        print(df.groupby(["arm", "status"]).size().to_string())
    return 0


def _cmd_sectors(args) -> int:
    from .sectors import report, select

    if args.action == "build":
        select.build(root=args.results, extension=not args.no_extension, N_values=args.N, force=args.force)
        return 0
    if args.action == "tables":
        t = select.build_tables(args.results)
        print(f"{len(t['runs'])} GA runs, {len(t['sectors'])} (sector, instance) rows")
        return 0
    if args.action == "report":
        if args.no_write:
            print(report.markdown(args.results))
        else:
            print(f"written to {report.write(args.results)}", file=sys.stderr)
        return 0
    if args.action == "pybind":
        from .sectors import pybind_check as pc
        from .store.paths import sectors_dir

        if not args.module_dir or not args.label:
            print("pybind needs --module-dir and --label", file=sys.stderr)
            return 2
        out = sectors_dir(args.results)
        out.mkdir(parents=True, exist_ok=True)
        Ns = tuple(args.N) if args.N else (4, 5, 6, 7)
        df = pc.validate(args.module_dir, args.label, rules=tuple(args.rules), N_values=Ns, root=args.results)
        if not df.empty:
            df.to_parquet(out / f"pybind_{args.label}.parquet", index=False)
        if args.seed_study:
            import pandas as pd

            rows = []
            for rule in args.rules:
                for iid in args.seed_insts:
                    for K in (12, 24):
                        rows.append(pc.seed_study(args.module_dir, [iid], rule, K, args.seed_study, args.results))
            sd = pd.concat([r for r in rows if not r.empty], ignore_index=True) if rows else None
            if sd is not None and not sd.empty:
                sd.to_parquet(out / f"pybind_seedstudy_{args.label}.parquet", index=False)
                print(sd.groupby(["inst_id", "rule", "K"])[["cpp_eq_bf", "numpy_eq_bf"]].mean().to_string())
        return 0
    raise AssertionError(args.action)


def _cmd_mixer(args) -> int:
    from .compile import report

    if args.action == "counts":
        df = report.write_counts(args.results)
        print(f"{len(df)} (cell, instance, order) rows")
        return 0
    if args.action == "c1":
        orders = tuple(args.orders) if args.orders else ("lex", "rank")
        sym = (False, True)
        df = report.write_c1(args.results, engine=args.engine, decomposed_n_max=args.decomposed_n_max,
                             ring_orders=orders, symmetrized=sym, limit=args.limit,
                             log=lambda m: print(m, file=sys.stderr, flush=True))
        print(f"{len(df)} checks; max leakage {df.leakage.max():.2e}, max block error {df.block_err.max():.2e}")
        ok = bool((df.leakage <= 1e-13).all() and (df.block_err <= 1e-12).all())
        return 0 if ok else 1
    if args.action == "time":
        from .compile import timing

        out = timing.write(args.results, n_inst=args.n_inst, repeats=args.repeats, builder=not args.no_builder)
        for r in out["rows"]:
            print(f"{r['cell']:<22} {r['inst_id']} {r['engine']:<7} L={r['L']:<2} gates={r['n_gates']:>6} "
                  f"build={r['build_s']:.3f}s first={r['first_observe_s']:.3f}s "
                  f"observe={1e3 * r['observe_median_s']:.2f}ms get_state={1e3 * r['get_state_median_s']:.1f}ms")
        return 0
    if args.action == "report":
        if args.no_write:
            print(report.markdown(args.results))
        else:
            print(f"written to {report.write(args.results)}", file=sys.stderr)
        return 0
    raise AssertionError(args.action)


def _cmd_arms(args) -> int:
    from .arms import checks

    log = lambda m: print(m, file=sys.stderr, flush=True)   # noqa: E731
    if args.action == "smoke":
        df = checks.smoke(args.results, log=log)
        print(f"{len(df)} runs, {int(df.valid.sum())} valid")
        return 0 if bool(df.valid.all()) else 1
    if args.action == "anneal":
        df = checks.anneal(args.results, log=log)
        v = checks.anneal_verdict(df)
        print(v.to_string(index=False))
        return 0 if bool(v["pass"].all()) else 1
    if args.action == "repro":
        df = checks.repro(args.results, log=log)
        print(df[["arm", "e", "ar_f", "ar2_completed", "delta", "iterations"]].to_string(index=False))
        return 0
    if args.action == "time":
        from .arms import timing
        timing.run_all(log=log)
        return 0
    if args.action == "report":
        from .arms import report
        if args.no_write:
            print(report.markdown(args.results))
        else:
            print(f"written to {report.write(args.results)}", file=sys.stderr)
        return 0
    if args.action == "run":
        from .arms.qaoa import A0, A1
        from .arms.ramp import A2
        arm = {"A0": A0, "A1": A1, "A2p": lambda: A2("penalty"), "A2c": lambda: A2("confined")}[args.arm]()
        cell = None if args.arm in ("A0", "A2p") else {"connectivity": args.conn, "rule": args.rule, "K": args.K}
        kw = {"lam": args.lam} if args.arm in ("A0", "A2p") else {}
        if args.arm in ("A0", "A1"):
            kw["restart"] = args.restart
        else:
            kw["schedule_tag"] = args.schedule
        r = arm.run(args.inst, cell, args.effort, root=args.results, **kw)
        print(json.dumps({k: v for k, v in r.record.items() if k.startswith(("run_id", "status", "metric_"))},
                         indent=1))
        return 0
    raise AssertionError(args.action)


def _cmd_metrics(args) -> int:
    log = lambda m: print(m, file=sys.stderr, flush=True)   # noqa: E731
    if args.action == "finalize":
        from .metrics.postrun import finalize_all
        from .sim import backend
        busy = backend.gpu_compute_pids()
        if busy:
            print(f"GPU busy (compute PIDs {busy}); one GPU process at a time", file=sys.stderr)
            return 3
        out = finalize_all(args.results, arms=args.arms, limit=args.limit, force=args.force,
                           log=log if args.verbose else None)
        print(json.dumps(out))
        return 0 if out["failed"] == 0 else 1
    if args.action == "checks":
        from .metrics import checks
        out = checks.run_all(args.results, log=log)
        print(json.dumps({"example_max_error": out["example"]["max_error"], "d1_ok": out["d1"]["all_ok"],
                          "d2_verdicts_ok": out["d2"]["verdicts_ok"], "c1_ok": out["c1"]["all_ok"],
                          "simdiff_max_F_diff": out["simdiff"]["max_F_diff"]}, indent=1))
        return 0
    if args.action == "report":
        from .metrics import report
        if args.no_write:
            print(report.markdown(args.results))
        else:
            print(f"written to {report.write(args.results)}", file=sys.stderr)
        return 0
    raise AssertionError(args.action)


def _cmd_aggregate(args) -> int:
    from .metrics.aggregate import aggregate
    df = aggregate(args.results, rebuild_all=args.rebuild, log=lambda m: print(m, file=sys.stderr, flush=True))
    if not df.empty:
        bad = df[df["anomalies"] != ""]
        print(f"{len(df)} rows; {len(bad)} with anomalies")
        if len(bad):
            print(bad.groupby(["arm", "anomalies"]).size().to_string())
    return 0


def _cmd_stats(args) -> int:
    from .stats import d1
    from .store.paths import tables_dir
    out = tables_dir(args.results)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "gbar":
        t = d1.gbar_table(root=args.results)
        t.to_parquet(out / "gbar.parquet", index=False)
        print(t.pivot(index="N", columns="arm", values="gbar").to_string())
        return 0
    if args.action == "d1":
        lam = {int(k): float(v) for k, v in (x.split("=") for x in (args.lam_star or []))} or None
        curves = d1.curves_from_store(args.results, d1.d1_spec(lam, args.ring_order))
        if curves.empty:
            print("no runs match the D1 filters")
            return 0
        gp = out / "gbar.parquet"
        if gp.exists():
            import pandas as pd
            g = pd.read_parquet(gp)
            gbar = {(int(r.N), r.arm): float(r.gbar) for r in g.itertuples()}
        else:
            gbar = {}
        res = d1.run_d1(curves, gbar)
        res["pairs"].to_parquet(out / "d1.parquet", index=False)
        if not res["choices"].empty:
            res["choices"].to_parquet(out / "d1_choices.parquet", index=False)
        p = res["pairs"]
        print(p.groupby(["N", "pairing", "family", "outcome"]).size().to_string())
        return 0
    raise AssertionError(args.action)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gsp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="group", required=True)

    pi = sub.add_parser("instances", help="freeze / verify / tabulate the instance set (PLAN §1.1)")
    pi.add_argument("action", choices=["freeze", "verify", "table"])
    pi.add_argument("--results", default=None, help="results root (default GSP/results or $GSP_RESULTS)")
    pi.add_argument("--dataset", default=None, help="dataset dir (default <repo>/dataset or $GSP_DATASET_DIR)")
    pi.add_argument("--N", type=int, nargs="+", default=None, help="freeze only these N (tests)")
    pi.add_argument("--draws", type=int, default=None, help="accepted draws per N (tests; default 30)")
    pi.add_argument("--deep", action="store_true", help="verify: rebuild in memory and compare")
    pi.add_argument("--replace-set", action="store_true",
                    help="freeze: replace the frozen set after a deliberate, logged rule change (surviving "
                         "ids must stay byte-identical; dropped ids' files are removed)")
    pi.add_argument("--no-write", action="store_true", help="table: print only")
    pi.set_defaults(func=_cmd_instances)

    ps = sub.add_parser("sectors", help="GA sectors, BF reference, controllability, report (PLAN §5 S2)")
    ps.add_argument("action", choices=["build", "tables", "report", "pybind"])
    ps.add_argument("--results", default=None, help="results root (default GSP/results or $GSP_RESULTS)")
    ps.add_argument("--N", type=int, nargs="+", default=None, help="only these N")
    ps.add_argument("--no-extension", action="store_true", help="build: skip the N = 8-10 timing runs")
    ps.add_argument("--force", action="store_true", help="build: re-run jobs whose files are current")
    ps.add_argument("--no-write", action="store_true", help="report: print only")
    ps.add_argument("--module-dir", default=None, help="pybind: directory holding a ga_solver build")
    ps.add_argument("--label", default=None, help="pybind: name of that build in the report")
    ps.add_argument("--rules", nargs="+", default=["violation", "objective"])
    ps.add_argument("--seed-study", type=int, default=0, help="pybind: seeds per instance (0 = none)")
    ps.add_argument("--seed-insts", nargs="+", default=["N05e008q1.5", "N06e000q1.5", "N07e000q1.5"],
                    help="pybind: instances of the seed study")
    ps.set_defaults(func=_cmd_sectors)

    pm = sub.add_parser("mixer", help="compiled preserving mixer: counts, C1 operator check, timing, report (S3)")
    pm.add_argument("action", choices=["counts", "c1", "time", "report"])
    pm.add_argument("--results", default=None)
    pm.add_argument("--engine", choices=["numpy", "cudaq", "builder"], default="numpy",
                    help="c1: numpy | cudaq (interpreter kernel) | builder (PLAN §2.4 builder kernels, slow JIT)")
    pm.add_argument("--decomposed-n-max", type=int, default=10,
                    help="c1: also run the explicit (iii) circuit at n <= this (0 = never)")
    pm.add_argument("--limit", type=int, default=None, help="c1: only the first LIMIT sectors (tests)")
    pm.add_argument("--orders", nargs="+", choices=["lex", "rank"], default=None,
                    help="c1: ring orders (default lex rank; symmetrized lex is always added)")
    pm.add_argument("--n-inst", type=int, default=3, help="time: instances per cell")
    pm.add_argument("--repeats", type=int, default=10, help="time: steady-state observe calls per circuit")
    pm.add_argument("--no-builder", action="store_true", help="time: interpreter engine only")
    pm.add_argument("--no-write", action="store_true", help="report: print only")
    pm.set_defaults(func=_cmd_mixer)

    pa = sub.add_parser("arms", help="arms A0 / A1 / A2: smoke run, anneal check, reproduction, timing, report (S4)")
    pa.add_argument("action", choices=["smoke", "anneal", "repro", "time", "report", "run"])
    pa.add_argument("--results", default=None)
    pa.add_argument("--no-write", action="store_true", help="report: print only")
    pa.add_argument("--arm", choices=["A0", "A1", "A2p", "A2c"], default="A1", help="run: the arm")
    pa.add_argument("--inst", default=None, help="run: inst_id")
    pa.add_argument("--effort", type=int, default=5, help="run: depth L (A0/A1) or ramp depth p (A2)")
    pa.add_argument("--K", type=int, default=12)
    pa.add_argument("--rule", default="violation")
    pa.add_argument("--conn", default="ring")
    pa.add_argument("--lam", type=float, default=None, help="run: lambda of a penalty arm")
    pa.add_argument("--restart", type=int, default=0)
    pa.add_argument("--schedule", default="primary")
    pa.set_defaults(func=_cmd_arms)

    pmt = sub.add_parser("metrics", help="post-run step, S5 checks and report (PLAN §1.7, §5 S5)")
    pmt.add_argument("action", choices=["finalize", "checks", "report"])
    pmt.add_argument("--results", default=None)
    pmt.add_argument("--arms", nargs="+", default=None, help="finalize: only these arms")
    pmt.add_argument("--limit", type=int, default=None, help="finalize: at most this many new runs")
    pmt.add_argument("--force", action="store_true", help="finalize: redo runs that already have the files")
    pmt.add_argument("--verbose", action="store_true", help="finalize: one line per run")
    pmt.add_argument("--no-write", action="store_true", help="report: print only")
    pmt.set_defaults(func=_cmd_metrics)

    pag = sub.add_parser("aggregate", help="results/tables/metrics.parquet from every done run (S5)")
    pag.add_argument("--results", default=None)
    pag.add_argument("--rebuild", action="store_true", help="recompute every row (default: incremental)")
    pag.set_defaults(func=_cmd_aggregate)

    pst = sub.add_parser("stats", help="Rule D1 inputs and tables (S5)")
    pst.add_argument("action", choices=["gbar", "d1"])
    pst.add_argument("--results", default=None)
    pst.add_argument("--lam-star", nargs="+", default=None, help="d1: lambda*(N) as N=LAM (S9); default: no filter")
    pst.add_argument("--ring-order", default="lex", choices=["lex", "rank"], help="d1: A1 / A2c ring order (D-9)")
    pst.set_defaults(func=_cmd_stats)

    px = sub.add_parser("index", help="rebuild the run registry from every run.json")
    px.add_argument("--results", default=None)
    px.set_defaults(func=_cmd_index)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
