"""
Merge results from split/parallel runs (made with `-post <postfix>`) back into the main
experiment folder, keeping the exact same file formats.

For every `exp_*` subfolder inside each `<root>_<postfix>` directory:
  - report_*.csv       -> concatenated into <root>/<sub>/report_*.csv, de-duplicated on the
                          key columns (Assets, Layer, Exp, Seed, Point, Boost - whichever
                          exist), split-run rows win over existing rows.
  - expectation_*.npz  -> keys merged into <root>/<sub>/expectation_*.npz, split-run keys
                          overwrite existing ones.

Example:
    python merge_split_results.py -root experiments_approx_Q2_LR_1.5_3.0_S1.0_W0.01_Jh \
        -post job1 job2 job3 job4 job5 job6
"""
import os
import argparse
import numpy as np
import pandas as pd

KEY_COLS = ["Assets", "Layer", "Exp", "Seed", "Point", "Boost"]


def merge_csv(src, dst, dry_run=False):
    df_src = pd.read_csv(src)
    if os.path.exists(dst):
        df_dst = pd.read_csv(dst)
        df = pd.concat([df_dst, df_src], ignore_index=True)
        n_before = df_dst.shape[0]
    else:
        df = df_src.copy()
        n_before = 0
    keys = [c for c in KEY_COLS if c in df.columns]
    df = df.drop_duplicates(subset=keys, keep="last")
    sort_cols = [c for c in ["Assets", "Layer", "Exp", "Seed", "Point"] if c in df.columns]
    df.sort_values(by=sort_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if not dry_run:
        df.to_csv(dst, index=False)
    print(f"  [csv] {os.path.basename(dst)}: {n_before} -> {df.shape[0]} rows (+{df_src.shape[0]} from split)")


def merge_npz(src, dst, dry_run=False):
    d_src = dict(np.load(src))
    if os.path.exists(dst):
        d = dict(np.load(dst))
        n_before = len(d)
    else:
        d = {}
        n_before = 0
    n_new = sum(1 for k in d_src if k not in d)
    d.update(d_src)
    if not dry_run:
        np.savez_compressed(dst, **d)
    print(f"  [npz] {os.path.basename(dst)}: {n_before} -> {len(d)} keys (+{n_new} new, {len(d_src) - n_new} overwritten)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge split (-post) experiment results back into the main folder")
    parser.add_argument("--root_dir", "-root", type=str, required=True,
                        help="Main root directory (WITHOUT postfix), e.g. experiments_approx_Q2_LR_1.5_3.0_S1.0_W0.01_Jh")
    parser.add_argument("--postfix", "-post", nargs="+", type=str, required=True,
                        help="List of postfixes to merge, e.g. -post job1 job2 job3")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Only print what would be merged, do not write anything")
    args = parser.parse_args()

    root = args.root_dir.rstrip("/")
    for post in args.postfix:
        src_root = f"{root}_{post}"
        if not os.path.isdir(src_root):
            print(f"!! {src_root} does not exist, skipping")
            continue
        print(f"== Merging {src_root} -> {root}")
        for sub in sorted(os.listdir(src_root)):
            src_sub = os.path.join(src_root, sub)
            if not os.path.isdir(src_sub):
                continue
            dst_sub = os.path.join(root, sub)
            if not args.dry_run:
                os.makedirs(dst_sub, exist_ok=True)
            print(f"  -- {sub}")
            for fname in sorted(os.listdir(src_sub)):
                src_f = os.path.join(src_sub, fname)
                dst_f = os.path.join(dst_sub, fname)
                if fname.startswith("report_") and fname.endswith(".csv"):
                    merge_csv(src_f, dst_f, args.dry_run)
                elif fname.startswith("expectation_") and fname.endswith(".npz"):
                    merge_npz(src_f, dst_f, args.dry_run)
                else:
                    print(f"  [??] skipping unknown file {fname}")
    print("Done." + (" (dry run, nothing written)" if args.dry_run else ""))
