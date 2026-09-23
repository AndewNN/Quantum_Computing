"""Command line: `gsp <group> <command>` or `python -m gsp.cli ...`.

S1:  gsp instances freeze [--N 4 5 ...] [--draws 30] [--results DIR]
     gsp instances verify [--deep] [--results DIR]
     gsp instances table  [--no-write] [--results DIR]
     gsp index            [--results DIR]     (rebuild results/index/registry.parquet)
Later sessions add sectors, plan, run, aggregate, report, missing, selfcheck.
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
                  draws_per_n=args.draws or DRAWS_PER_N)
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
    pi.add_argument("--no-write", action="store_true", help="table: print only")
    pi.set_defaults(func=_cmd_instances)

    px = sub.add_parser("index", help="rebuild the run registry from every run.json")
    px.add_argument("--results", default=None)
    px.set_defaults(func=_cmd_index)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
