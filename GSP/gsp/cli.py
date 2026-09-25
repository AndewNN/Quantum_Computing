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
                                      (S7: --arm A3 | A3d --lam 0.005 [--n-steps 10] runs VarQITE)
                                      (S8: --arm A4 --effort 5 [--K 12 --rule violation] / --arm A6 --lam 0.005
                                       --effort 5 [--step-units normalized|plan] runs DB-QITE for 5 steps)
     (scripts/s4_legacy_checks.py: the legacy-equivalence numbers, results/tables/s4_legacy.json)
S5:  gsp metrics finalize [--arms A0 ...] [--limit N] [--force]
                                      (GPU: samples.npz + postrun.json for the stored runs that lack them)
     gsp metrics checks               (the S5 evidence -> results/tables/s5_checks.json)
     gsp metrics report [--no-write]  (reports/metrics.md)
     gsp aggregate [--rebuild]        (results/tables/metrics.parquet, one row per done run; incremental)
     gsp stats gbar                   (results/tables/gbar.parquet: gbar_arm(N) of Rule D1 from the frozen set)
     gsp stats d1 [--lam-star 4=0.005 5=...] [--ring-order lex]
                                      (Rule D1 over the stored runs -> results/tables/d1.parquet, d1_choices.parquet)
S6:  gsp plan [filters] [--lam-star 4=0.005 ...] [--gate-matched 4:5=45 ...] [--recursion-cap A4:4=12 ...]
              [--optional | --pilot] [--name NAME | --out FILE] [--dry-run] [--order draw|arm]
                                      (the OFAT envelope -> a queue file; counts per arm and cell first. Filters:
                                       --arms --N --draws K --q --axes --efforts --restarts R --schedules --tags.
                                       An unfiltered queue is written only from an `approved` envelope)
     gsp run --queue FILE [--max-runs N] [--no-retry-failed] [--steal] [--no-aggregate] [--runs-root DIR]
                                      (sequential, resumable; SIGTERM/SIGINT = clean stop, SIGUSR1 = stop after
                                       the current run; progress in results/logs/queue_NAME.progress.json)
     gsp progress [NAME ...]          (the heartbeat / progress files of the queues)
     gsp missing [--queue FILE | plan filters] [--list] [--out FILE] [--json]
                                      (which cells / instances / efforts lack done runs, in one call)
     gsp merge --src DIR [--dry-run]  (fold a pulled remote results/runs shard in; scripts/remote/pull.sh)
     gsp report [--write]             (stub: runs per arm x status, metrics rows, queues; S15 builds it out)
     gsp index [--check]              (rebuild the registry; --check validates every run.json)
S8:  gsp selfcheck --record [--force] | --compare [--json]   (CUDA-Q migration acceptance test, PLAN §3.5; GPU)
     gsp arms run --arm A4 | A6 ...   (DB-QITE; scripts/s8_dbqite_checks.py: example / C1 / smoke / timing / report)
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
    if getattr(args, "check", False) and not df.empty:
        from .store.paths import runs_dir
        from .store.records import read_record, validate_record
        bad = []
        dup = df["run_id"].duplicated(keep=False)
        for r in df[dup].itertuples():
            bad.append(f"{r.run_dir}: run_id {r.run_id} appears in more than one arm directory")
        for r in df.itertuples():
            try:
                rec = read_record(runs_dir(args.results) / r.run_dir / "run.json")
                validate_record(rec)
                if r.run_dir != f"{rec['arm']}/{rec['run_id']}":
                    raise ValueError(f"directory is not {rec['arm']}/{rec['run_id']}")
            except Exception as exc:
                bad.append(f"{r.run_dir}: {exc}")
        print(f"check: {len(df)} records, {df['run_id'].nunique()} distinct run_ids, {len(bad)} problem(s)")
        for b in bad[:50]:
            print("  " + b)
        return 0 if not bad else 1
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
        from .arms.base import make_arm
        arm = make_arm(args.arm)
        penalty = args.arm in ("A0", "A2p", "A3", "A3d", "A6")
        conn = "adaptive" if args.arm == "A4" else args.conn
        cell = None if penalty else {"connectivity": conn, "rule": args.rule, "K": args.K}
        kw = {"lam": args.lam} if penalty else {}
        if args.arm in ("A0", "A1"):
            kw["restart"] = args.restart
        elif args.arm in ("A3", "A3d"):
            if args.n_steps is not None:
                kw["n_steps"] = args.n_steps
        elif args.arm in ("A4", "A6"):
            kw["step_units"] = args.step_units
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


# --- S6: plan / run / progress / missing / merge / report ---------------------------------------------------------
def _kv(items, what):
    out = {}
    for x in items or []:
        try:
            k, v = x.split("=")
        except ValueError:
            raise SystemExit(f"{what}: expected KEY=VALUE, got {x!r}")
        out[k] = v
    return out


def _plan_inputs(args):
    from .runner import plan as P
    env = P.load_envelope(args.envelope)
    lam = {int(k): float(v) for k, v in _kv(args.lam_star, "--lam-star").items()}
    gm = {}
    for k, v in _kv(args.gate_matched, "--gate-matched").items():
        N, L1 = k.split(":")
        gm.setdefault(int(N), {})[int(L1)] = int(v)
    rc = {}
    for k, v in _kv(args.recursion_cap, "--recursion-cap").items():
        a, N = k.split(":")
        rc.setdefault(a, {})[int(N)] = int(v)
    params = P.PlanParams.from_envelope(env, lam_star=lam or None, gate_matched=gm or None,
                                        recursion_cap=rc or None, ring_order=args.ring_order)
    t = lambda x, f=str: None if x is None else tuple(f(v) for v in x)   # noqa: E731
    flt = P.PlanFilter(arms=t(args.arms), N=t(args.N, int), draws=args.draws, q=t(args.q, float),
                       axes=t(args.axes), efforts=t(args.efforts, int), restarts=args.restarts,
                       schedules=t(args.schedules), tags=t(args.tags))
    return env, params, flt


def _add_plan_args(p):
    p.add_argument("--envelope", default=None, help="envelope yaml (default configs/envelope.yaml)")
    p.add_argument("--arms", nargs="+", default=None, help="only these arms (A0 A1 A2p A2c A3 A4 A6 A3d)")
    p.add_argument("--N", type=int, nargs="+", default=None)
    p.add_argument("--draws", type=int, default=None, help="the first K accepted draws per N (per cell subset)")
    p.add_argument("--q", type=float, nargs="+", default=None)
    p.add_argument("--axes", nargs="+", default=None, help="baseline K rule connectivity penalty")
    p.add_argument("--efforts", type=int, nargs="+", default=None, help="only these depths / ramp depths / caps")
    p.add_argument("--restarts", type=int, default=None, help="at most R restarts (r < R)")
    p.add_argument("--schedules", nargs="+", default=None, help="ramp schedules (primary secondary)")
    p.add_argument("--tags", nargs="+", default=None, help="main gate_matched")
    p.add_argument("--lam-star", nargs="+", default=None, help="lambda*(N) as N=LAM (S9); else from the envelope")
    p.add_argument("--gate-matched", nargs="+", default=None, help="L0 as N:L1=L0 (S9, PLAN §1.3)")
    p.add_argument("--recursion-cap", nargs="+", default=None, help="cap as ARM:N=K (S9, PLAN §1.8)")
    p.add_argument("--ring-order", default="lex", choices=["lex", "rank"], help="A1 / A2c / A4 ring order (D-9)")
    p.add_argument("--optional", action="store_true", help="include the optional tier (A3d)")
    p.add_argument("--pilot", action="store_true", help="the lambda pilot of PLAN §1.4 instead of the sweep")


def _cmd_plan(args) -> int:
    from .runner import plan as P
    env, params, flt = _plan_inputs(args)
    log = lambda m: print(m, file=sys.stderr, flush=True)   # noqa: E731
    full = P.counts(P.expand(env, params, P.PlanFilter(), optional=True, root=args.results))
    print(f"envelope {env['_path']} status={env.get('status')} harness={env.get('harness_version')} "
          f"ring_order={params.ring_order}")
    print(f"S9 inputs: lambda*={params.lam_star or 'unset'} gate-matched L0={params.gate_matched or 'unset'} "
          f"recursion cap={params.recursion_cap or 'unset'}")
    print("full grid (the envelope, no filters; optional tier marked): "
          + ", ".join(f"{a} {v['total']}" + (" [optional]" if v["tier"] != "main" else "")
                      for a, v in full["per_arm"].items())
          + f"; total {full['total']} (without the optional tier "
          + f"{sum(v['total'] for v in full['per_arm'].values() if v['tier'] == 'main')})")
    selected = args.pilot or flt.active() or args.optional
    specs = P.build_plan(env, params, flt, optional=args.optional, pilot=args.pilot, order=args.order,
                         resolve_ids=not args.dry_run, root=args.results, log=log)
    c = P.counts(specs)
    title = "lambda pilot (PLAN §1.4)" if args.pilot else ("selected subset" if flt.active() else "the plan")
    print()
    print(P.format_counts(c, env, f"{title}: {c['total']} runs, {c['runnable']} runnable"))
    if args.json:
        print(json.dumps(c, indent=1))
    if args.dry_run:
        return 0
    if not (args.out or args.name):
        print("\n(no --name / --out: nothing written)")
        return 0
    if not selected and env.get("status") != "approved":
        print(f"\nrefused: an unfiltered queue (the S10 sweep) is written only from an envelope with "
              f"status: approved (it is {env.get('status')!r}; PLAN §1.8)", file=sys.stderr)
        return 2
    path = args.out or P.default_queue_path(args.name, args.results)
    meta = {"envelope": env["_path"], "envelope_sha256": env["_sha256"], "envelope_status": env.get("status"),
            "params": params.as_dict(), "filters": flt.as_dict(), "optional": args.optional, "pilot": args.pilot,
            "order": args.order, "counts": c, "full_grid": {a: v["total"] for a, v in full["per_arm"].items()}}
    from .store.records import git_info
    meta["git_sha"], meta["git_dirty"] = git_info()
    side = P.write_queue(specs, path, meta)
    print(f"\nwrote {side['n_runs']} runnable specs to {path} ({side['n_placeholders']} placeholders not queued); "
          f"sidecar {P.sidecar_path(path)}")
    return 0


def _cmd_run(args) -> int:
    from .runner.queue import EXIT_BUSY, EXIT_USAGE, QueueBusy, run_queue
    from .runner.plan import PlanError
    try:
        out = run_queue(args.queue, root=args.results, out_root=args.runs_root, retry_failed=not args.no_retry_failed,
                        steal=args.steal, gpu=not args.no_gpu_guard, finalize=not args.no_finalize,
                        aggregate=not args.no_aggregate, heartbeat_s=args.heartbeat, max_runs=args.max_runs,
                        debug_pause_postrun=args.debug_pause_postrun, dynamic=args.dynamic, follow=args.follow,
                        poll_s=args.poll)
    except QueueBusy as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_BUSY
    except (PlanError, FileNotFoundError) as exc:
        print(f"bad queue: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(json.dumps(out))
    return int(out["exit_code"])


def _cmd_q(args) -> int:
    from .runner import qedit
    a = args.action
    if a == "ls":
        print(json.dumps(qedit.ls(args.queue, args.results, args.show), indent=1))
        return 0
    if a in ("rm", "top") and not args.match:
        print(f"q {a} needs --match", file=sys.stderr)
        return 2
    if a in ("add", "take") and not args.other:
        print(f"q {a} needs the other file", file=sys.stderr)
        return 2
    n = {"add": lambda: qedit.add(args.queue, args.other, top=args.top),
         "rm": lambda: qedit.rm(args.queue, args.match),
         "top": lambda: qedit.top(args.queue, args.match),
         "take": lambda: qedit.take(args.queue, args.other, match=args.match, tail=args.tail,
                                    out_root=args.results)}[a]()
    print(f"q {a}: {n} line(s)")
    return 0


def _cmd_tqe(args) -> int:
    from pathlib import Path
    from .runner import tqe
    speeds = {m.split("=")[0]: float(m.split("=")[1]) for m in args.machines}
    specs = tqe.all_specs(exps=tuple(args.exps), parts=tuple(args.parts or tqe.PARTS),
                          ramp=tuple(args.ramp) if args.ramp else tqe.RAMP_DEFAULT)
    summ = tqe.summary(specs)
    print(json.dumps(summ, indent=1))
    parts, hours = tqe.split(specs, speeds)
    print(f"{len(specs)} runs; est. hours per machine (likely): " + ", ".join(f"{m} {h:.1f}" for m, h in hours.items()))
    if args.dry_run:
        return 0
    specs = tqe.resolve_specs(specs, args.results)
    rid = {(s["label"], s["tier"]): s for s in specs}
    for m, lst in parts.items():
        out = Path(args.out_dir) / f"tqe_{m}.jsonl"
        tqe.write_lines([rid[(s["label"], s["tier"])] for s in lst], out)
        print(f"wrote {out}: {len(lst)} lines")
    return 0


def _cmd_progress(args) -> int:
    from .runner.queue import format_progress, read_progress
    from .store.paths import logs_dir
    names = args.names or sorted(p.name[len("queue_"):-len(".progress.json")]
                                 for p in logs_dir(args.results).glob("queue_*.progress.json"))
    if not names:
        print("no progress files")
        return 1
    for n in names:
        pr = read_progress(n, args.results)
        print(json.dumps(pr, indent=1) if args.json else format_progress(pr))
    return 0


def _cmd_missing(args) -> int:
    from .runner import missing as M
    from .runner import plan as P
    if args.queue:
        specs = P.read_queue(args.queue)
        src = f"queue {args.queue}"
    else:
        env, params, flt = _plan_inputs(args)
        specs = P.build_plan(env, params, flt, optional=args.optional, pilot=args.pilot, root=args.results)
        src = f"plan of {env['_path']} (status {env.get('status')}; filters {flt.as_dict() or 'none'})"
    out_root = args.runs_root or args.results
    df = M.coverage(specs, out_root)
    if args.json:
        print(M.dumps(M.as_json(df)))
    else:
        print(f"source: {src}")
        print(M.summarize(df, max_lines=args.max_lines))
        if args.list:
            print()
            print(M.list_lines(df))
    if args.out:
        n = M.write_missing_queue(df, args.out)
        print(f"wrote {n} runnable missing specs to {args.out}", file=sys.stderr)
    return 0 if (df["state"] == "done").all() else 1


def _cmd_merge(args) -> int:
    from .runner.merge import merge_runs
    from .store.index import build_index
    out = merge_runs(args.src, args.results, dry_run=args.dry_run,
                     log=(lambda m: print(m, file=sys.stderr)) if args.verbose else None)
    if not args.dry_run:
        df = build_index(args.results)
        out["indexed"] = len(df)
    print(json.dumps(out, indent=1))
    return 0 if not out["invalid"] else 1


def _cmd_report(args) -> int:
    from .runner import report
    if args.write:
        print(f"written to {report.write(args.results)}", file=sys.stderr)
    else:
        print(report.markdown(args.results))
    return 0


def _cmd_selfcheck(args) -> int:
    from .sim import backend, selfcheck
    log = lambda m: print(m, file=sys.stderr, flush=True)   # noqa: E731
    busy = backend.gpu_compute_pids()
    if busy:
        print(f"GPU busy (compute PIDs {busy}); one GPU process at a time", file=sys.stderr)
        return 3
    if args.record:
        try:
            path = selfcheck.record(args.fixture, force=args.force, root=args.results, log=log)
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"recorded {path}")
        return 0
    res = selfcheck.compare(args.fixture, root=args.results, log=log)
    if args.json:
        print(json.dumps(res, indent=1))
    print(f"selfcheck {res['running_version']} vs fixture {res['fixture_version']}: {res['n_cases']} cases, "
          f"{res['n_fail']} failed -> {'PASS' if res['passes'] else 'FAIL'}")
    return 0 if res["passes"] else 1


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
    pa.add_argument("--arm", choices=["A0", "A1", "A2p", "A2c", "A3", "A3d", "A4", "A6"], default="A1",
                    help="run: the arm")
    pa.add_argument("--step-units", choices=["normalized", "plan"], default="normalized",
                    help="run (A4 / A6): the DB-QITE step convention (default normalized = PLAN §1.5 as corrected in "
                         "S8b; plan = the S0b wording, a flag)")
    pa.add_argument("--n-steps", type=int, default=None, help="run (A3 / A3d): step cap for a smoke run (default 300)")
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
    px.add_argument("--check", action="store_true", help="validate every run.json (hash, directory, schema)")
    px.set_defaults(func=_cmd_index)

    ppl = sub.add_parser("plan", help="the OFAT envelope -> a queue file; counts per arm and cell (S6)")
    _add_plan_args(ppl)
    ppl.add_argument("--results", default=None, help="results root holding the instances / sectors")
    ppl.add_argument("--name", default=None, help="write results/queues/NAME.jsonl")
    ppl.add_argument("--out", default=None, help="write the queue here instead")
    ppl.add_argument("--order", default="draw", choices=["draw", "arm"])
    ppl.add_argument("--dry-run", action="store_true", help="counts only (no run_ids, nothing written)")
    ppl.add_argument("--json", action="store_true", help="also print the counts as JSON")
    ppl.set_defaults(func=_cmd_plan)

    prn = sub.add_parser("run", help="run a queue file: sequential, resumable, clean stop on SIGTERM/SIGINT (S6)")
    prn.add_argument("--queue", required=True)
    prn.add_argument("--results", default=None, help="results root holding the instances / sectors")
    prn.add_argument("--runs-root", default=None, help="results root the runs / logs are written to (default "
                                                       "--results)")
    prn.add_argument("--max-runs", type=int, default=None, help="run at most N new runs, then stop")
    prn.add_argument("--no-retry-failed", action="store_true", help="skip runs whose run.json says failed")
    prn.add_argument("--steal", action="store_true", help="re-run 'running' records written by another host")
    prn.add_argument("--no-finalize", action="store_true", help="skip the post-run step (not for production)")
    prn.add_argument("--no-aggregate", action="store_true", help="skip index + aggregate at the end")
    prn.add_argument("--heartbeat", type=float, default=30.0, help="progress-file heartbeat period (s)")
    prn.add_argument("--no-gpu-guard", action="store_true", help="do not refuse when the GPU is busy")
    prn.add_argument("--dynamic", action="store_true", help="re-read the queue file before every run (edit it live "
                     "with `gsp q`); file order = priority")
    prn.add_argument("--follow", action="store_true", help="with --dynamic: wait for new lines when exhausted")
    prn.add_argument("--poll", type=float, default=30.0, help="--follow poll period (s)")
    prn.add_argument("--debug-pause-postrun", type=float, default=0.0,
                     help="TEST ONLY: sleep S seconds between 'done' and the post-run step (kill tests)")
    prn.set_defaults(func=_cmd_run)

    pq = sub.add_parser("q", help="edit a queue file live (dynamic runner; TQE rerun)")
    pq.add_argument("action", choices=["ls", "add", "rm", "top", "take"])
    pq.add_argument("queue")
    pq.add_argument("other", nargs="?", default=None, help="add: source file; take: destination file")
    pq.add_argument("--match", default=None, help="substring of the raw JSON line")
    pq.add_argument("--tail", type=int, default=None, help="take: the last N pending lines")
    pq.add_argument("--top", action="store_true", help="add: prepend instead of append")
    pq.add_argument("--results", default=None, help="results root of the runs (states)")
    pq.add_argument("--show", type=int, default=10)
    pq.set_defaults(func=_cmd_q)

    ptq = sub.add_parser("tqe", help="TQE rerun: plan the Exp1-4 queues and split them across machines")
    ptq.add_argument("--exps", type=int, nargs="+", default=[1, 2, 3, 4])
    ptq.add_argument("--parts", nargs="+", default=None, help="default: every part, in priority order")
    ptq.add_argument("--ramp", type=float, nargs=2, default=None, metavar=("DBETA", "DGAMMA"))
    ptq.add_argument("--machines", nargs="+", default=["local=1.0"], help="name=relative speed (local 4080 = 1)")
    ptq.add_argument("--out-dir", required=True, help="where {name}.jsonl queue files are written")
    ptq.add_argument("--results", default=None, help="the TQE results root (default GSP_RESULTS; no frozen instances)")
    ptq.add_argument("--dry-run", action="store_true")
    ptq.set_defaults(func=_cmd_tqe)

    ppg = sub.add_parser("progress", help="show queue progress / heartbeat files (S6)")
    ppg.add_argument("names", nargs="*", help="queue names (default: every progress file)")
    ppg.add_argument("--results", default=None)
    ppg.add_argument("--json", action="store_true")
    ppg.set_defaults(func=_cmd_progress)

    pms = sub.add_parser("missing", help="which cells / instances / efforts lack done runs (S6)")
    pms.add_argument("--queue", default=None, help="a queue file (default: the plan from the filters below)")
    _add_plan_args(pms)
    pms.add_argument("--results", default=None)
    pms.add_argument("--runs-root", default=None)
    pms.add_argument("--list", action="store_true", help="one line per missing run")
    pms.add_argument("--out", default=None, help="write the runnable missing specs as a queue file")
    pms.add_argument("--json", action="store_true")
    pms.add_argument("--max-lines", type=int, default=40)
    pms.set_defaults(func=_cmd_missing)

    pmg = sub.add_parser("merge", help="fold a pulled remote results/runs shard into the store (S6)")
    pmg.add_argument("--src", required=True)
    pmg.add_argument("--results", default=None)
    pmg.add_argument("--dry-run", action="store_true")
    pmg.add_argument("--verbose", action="store_true")
    pmg.set_defaults(func=_cmd_merge)

    psc = sub.add_parser("selfcheck", help="CUDA-Q migration acceptance test: record / compare the fixture (§3.5, S8)")
    g = psc.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", action="store_true", help="run the set and write tests/fixtures/selfcheck_cudaq-<v>.json")
    g.add_argument("--compare", action="store_true", help="run the set again and compare with the fixture")
    psc.add_argument("--fixture", default=None, help="fixture path (record: default by the running version; "
                     "compare: default selfcheck_cudaq-0.15.1.json)")
    psc.add_argument("--force", action="store_true", help="record: replace an existing fixture")
    psc.add_argument("--results", default=None)
    psc.add_argument("--json", action="store_true", help="compare: print the full result as JSON")
    psc.set_defaults(func=_cmd_selfcheck)

    prp = sub.add_parser("report", help="harness report stub (S15 builds it out)")
    prp.add_argument("--results", default=None)
    prp.add_argument("--write", action="store_true", help="write reports/runs.md")
    prp.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
