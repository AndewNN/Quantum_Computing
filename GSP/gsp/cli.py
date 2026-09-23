"""Command line: `gsp <group> <command>` or `python -m gsp.cli ...`.

S1:  gsp instances freeze [--N 4 5 ...] [--draws 30] [--results DIR] [--replace-set]
     gsp instances verify [--deep] [--results DIR]
     gsp instances table  [--no-write] [--results DIR]
     gsp index            [--results DIR]     (rebuild results/index/registry.parquet)
S2:  gsp sectors build    [--N ...] [--no-extension] [--force] [--results DIR]
     gsp sectors tables | report [--no-write]
     gsp sectors pybind   --module-dir DIR --label NAME [--rules ...] [--N ...] [--seed-study N_SEEDS]
Later sessions add plan, run, aggregate, report, missing, selfcheck.
"""

from __future__ import annotations

from . import _threads  # noqa: F401  (pins BLAS/OpenMP threads before numpy loads)

import argparse
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

    px = sub.add_parser("index", help="rebuild the run registry from every run.json")
    px.add_argument("--results", default=None)
    px.set_defaults(func=_cmd_index)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
