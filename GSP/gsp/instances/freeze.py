"""`gsp instances freeze | verify | table` (PLAN §1.1, §3.2, §5 S1).

freeze  For N = 4..10, consume seed indices e = 0, 1, 2, ... in order; a draw is accepted iff
        |F_eps| >= K_req = 12 (every N, S1b); stop at 30 accepted draws per N. `k24_eligible` marks the
        accepted draws with |F_eps| >= 24, the only draws of the K = 24 cells (N = 5, 6). Every scanned draw (accepted or
        rejected, with its |F_eps|) and every seed goes into seed_table.csv. For each accepted
        draw and q in {1.0, 1.5, 3.0}: inst_{inst_id}.npz and rulers_{inst_id}.npz. Then
        instances.parquet and CHECKSUMS (sha256 per file, plus the dataset sources).
        Files are byte-reproducible; an existing file is never overwritten with different bytes
        (a changed instance needs a new id).
verify  Checks every file against CHECKSUMS and the dataset sources; `deep=True` also rebuilds
        everything in memory and compares arrays and tables exactly.
table   The |F_eps| table per (N, K in {6, 8, 12, 24}) and the rejections per N, as markdown
        (written to reports/instances.md).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .._version import HARNESS_VERSION
from ..store.io import atomic_write_bytes, load_npz, npz_bytes, sha256_bytes, sha256_file
from ..store.paths import (checksums_path, instances_dir, instances_parquet_path, inst_path,
                           reports_dir, rulers_path, seed_table_path)
from .data import COV_FILE, RET_FILE, Market, default_dataset_dir, load_market
from .draws import (DRAWS_PER_N, EPS, GA_RULES, MAX_SEED_INDEX, N_RESTARTS, N_VALUES, Q_VALUES,
                    K24, K24_CELL_N, Draw, ga_seed, inst_id, k24_eligible, k_req, make_draw,
                    restart_seed)
from .encode import Encoding, encode, po_normalize, ret_cov_to_QUBO
from .rulers import TIE_RTOL, Band, Rulers, band, rulers_from_band

K_TABLE = (6, 8, 12, 24)


class FreezeConflict(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# building (pure, in memory)
# ----------------------------------------------------------------------------------------------

@dataclass
class DrawScan:
    draw: Draw
    band: Band
    accepted: bool
    rank: int | None           # 1..30 among accepted draws of this N


@dataclass
class Frozen:
    seed_table: pd.DataFrame
    instances: pd.DataFrame
    files: dict = field(default_factory=dict)   # file name -> bytes (npz)
    sources: dict = field(default_factory=dict)


def draw_band(d: Draw) -> Band:
    """The band of a draw (q-independent): the old QU_lamb at lambda = 1 (PO_new_ApproxRatio.py:588)."""
    P_bb, ret_bb, cov_bb, n, _, _ = po_normalize(d.B, d.P, d.ret, d.cov)
    QU_pen = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, 1.0, 0.0)
    return band(QU_pen, P_bb, EPS)


def scan_size(market: Market, N: int, n_accept: int = DRAWS_PER_N, log=None) -> list[DrawScan]:
    out, accepted, e = [], 0, 0
    need = k_req(N)
    while accepted < n_accept:
        if e >= MAX_SEED_INDEX:
            raise RuntimeError(f"N={N}: only {accepted} accepted draws in {MAX_SEED_INDEX} seeds")
        d = make_draw(market, N, e)
        b = draw_band(d)
        ok = b.size >= need
        accepted += ok
        out.append(DrawScan(d, b, ok, accepted if ok else None))
        if log and not ok:
            log(f"  N={N} e={e}: rejected, |F_eps| = {b.size} < K_req = {need}")
        e += 1
    return out


def _s(x) -> np.ndarray:
    return np.array(x)


def instance_arrays(d: Draw, enc: Encoding, iid: str, sources: dict) -> dict:
    H, Pn = enc.H_obj, enc.Pen
    return {
        "inst_id": _s(iid), "draw_id": _s(d.draw_id), "harness_version": _s(HARNESS_VERSION),
        "N": _s(d.N), "e": _s(d.e), "q": _s(enc.q), "n": _s(enc.n), "eps": _s(EPS),
        "draw_seed": _s(d.seed),
        "asset_idx": d.asset_idx, "asset_idx_raw": d.asset_idx_raw,
        "tickers": d.tickers.astype("U"), "names": d.names.astype("U"),
        "P": d.P, "ret": d.ret, "cov": d.cov,
        "w": _s(d.w), "B_min": _s(d.B_min), "B_max": _s(d.B_max), "B": _s(d.B),
        "P_bb": enc.P_bb, "ret_bb": enc.ret_bb, "cov_bb": enc.cov_bb,
        "n_max": np.asarray(enc.n_max, dtype=np.int64), "C": enc.C,
        "QU_obj": enc.QU_obj, "QU_pen": enc.QU_pen,
        "obj_const": _s(H.const), "obj_h": H.h, "obj_J": H.J, "obj_has_h": H.has_h, "obj_has_J": H.has_J,
        "pen_const": _s(Pn.const), "pen_h": Pn.h, "pen_J": Pn.J, "pen_has_h": Pn.has_h, "pen_has_J": Pn.has_J,
        "boost_obj": _s(float(enc.boost_obj)),
        "source_cov_sha256": _s(sources[COV_FILE]), "source_ret_sha256": _s(sources[RET_FILE]),
    }


def rulers_arrays(r: Rulers) -> dict:
    return {
        "n": _s(r.n), "eps": _s(r.eps), "tie_rtol": _s(TIE_RTOL),
        "band_idx": r.band_idx.astype(np.uint32), "band_pen": r.band_pen, "f_band": r.f_band,
        "E_min": _s(r.E_min), "E_max": _s(r.E_max),
        "xstar_idx": r.xstar_idx.astype(np.uint32), "top10_idx": r.top10_idx.astype(np.uint32),
        "top10_size": _s(r.top10_size), "gap_band": _s(r.gap_band), "n_distinct": _s(r.n_distinct),
        "F_size": _s(r.F_size), "f_all_min": _s(r.f_all_min), "f_all_max": _s(r.f_all_max),
        "n_direct_disagree": _s(r.n_direct_disagree), "n_near_boundary": _s(r.n_near_boundary),
    }


def _fx(x: float) -> str:
    return repr(float(x))


def seed_row(s: DrawScan) -> dict:
    d, b = s.draw, s.band
    row = {
        "N": d.N, "e": d.e, "draw_id": d.draw_id, "n": b.n, "draw_seed": d.seed,
        "F_eps": b.size, "K_req": k_req(d.N), "accepted": bool(s.accepted),
        "accept_rank": s.rank if s.rank is not None else "",
        "k24_eligible": k24_eligible(b.size, s.accepted),
        "asset_idx": " ".join(str(int(i)) for i in d.asset_idx),
        "asset_idx_raw": " ".join(str(int(i)) for i in d.asset_idx_raw),
        "tickers": " ".join(d.tickers),
        "w": _fx(d.w), "B_min": _fx(d.B_min), "B_max": _fx(d.B_max), "B": _fx(d.B),
    }
    for r in range(N_RESTARTS):
        row[f"restart_seed_r{r}"] = restart_seed(d.N, d.e, r)
    for k, rule in enumerate(GA_RULES):
        row[f"ga_seed_{rule}"] = ga_seed(d.N, d.e, k)
    row["n_direct_disagree"] = b.n_direct_disagree
    row["n_near_boundary"] = b.n_near_boundary
    return row


def build(N_values=N_VALUES, draws_per_n: int = DRAWS_PER_N, q_values=Q_VALUES,
          dataset_dir=None, log=print) -> Frozen:
    market = load_market(dataset_dir)
    seed_rows, inst_rows, files = [], [], {}
    for N in N_values:
        t0 = time.perf_counter()
        scans = scan_size(market, N, draws_per_n, log=log)
        for s in scans:
            seed_rows.append(seed_row(s))
            if not s.accepted:
                continue
            d = s.draw
            for q in q_values:
                iid = inst_id(d.N, d.e, q)
                enc = encode(d.B, d.P, d.ret, d.cov, q)
                rul = rulers_from_band(s.band, enc.QU_obj)
                fi, fr = inst_path(iid, "/").name, rulers_path(iid, "/").name
                files[fi] = npz_bytes(instance_arrays(d, enc, iid, market.sources))
                files[fr] = npz_bytes(rulers_arrays(rul))
                inst_rows.append({
                    "inst_id": iid, "draw_id": d.draw_id, "N": d.N, "e": d.e, "q": float(q),
                    "n": enc.n, "eps": EPS, "B": d.B, "F_size": rul.F_size,
                    "E_min": rul.E_min, "E_max": rul.E_max, "n_xstar": int(rul.xstar_idx.size),
                    "top10_size": rul.top10_size, "gap_band": rul.gap_band,
                    "n_distinct": rul.n_distinct, "f_all_min": rul.f_all_min,
                    "f_all_max": rul.f_all_max, "boost_obj": float(enc.boost_obj),
                    "accept_rank": s.rank,
                    "k24_eligible": k24_eligible(rul.F_size, True),
                    "inst_file": fi, "rulers_file": fr,
                    "inst_sha256": sha256_bytes(files[fi]), "rulers_sha256": sha256_bytes(files[fr]),
                    "harness_version": HARNESS_VERSION,
                })
        n_rej = sum(not s.accepted for s in scans)
        if log:
            log(f"N={N}: scanned {len(scans)} draws, accepted {draws_per_n}, rejected {n_rej} "
                f"({time.perf_counter() - t0:.1f} s)")
    seed = pd.DataFrame(seed_rows)
    inst = pd.DataFrame(inst_rows)
    return Frozen(seed_table=seed, instances=inst, files=files, sources=market.sources)


def seed_table_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode()


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------
# writing, verifying
# ----------------------------------------------------------------------------------------------

def _all_files(fz: Frozen) -> dict:
    out = dict(fz.files)
    out[seed_table_path("/").name] = seed_table_bytes(fz.seed_table)
    out[instances_parquet_path("/").name] = parquet_bytes(fz.instances)
    return out


def checksums_text(files: dict, sources: dict) -> str:
    lines = [f"# GSP frozen instances; harness_version {HARNESS_VERSION}",
             "# format: <sha256>  <file in results/instances>; '# source' lines hash the dataset"]
    for name, digest in sorted(sources.items()):
        lines.append(f"# source dataset/{name} sha256 {digest}")
    for name in sorted(files):
        lines.append(f"{sha256_bytes(files[name])}  {name}")
    return "\n".join(lines) + "\n"


def write(fz: Frozen, root=None, log=print, replace_set: bool = False) -> dict:
    """Write a built set. Instance/ruler files are never overwritten with different bytes.

    Default (strict): only an identical re-freeze or a freeze into an empty directory succeeds.
    `replace_set=True` (used once, in S1b, when the acceptance rule changed before any downstream
    use): surviving ids must still be byte-identical (else FreezeConflict, nothing touched); new
    ids are written; the files of ids the new set no longer contains are deleted; seed_table.csv,
    instances.parquet and CHECKSUMS are replaced.
    """
    d = instances_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    files = _all_files(fz)
    tables = {seed_table_path("/").name, instances_parquet_path("/").name}
    conflicts = []
    for name, data in files.items():
        p = d / name
        if p.exists() and p.read_bytes() != data and not (replace_set and name in tables):
            conflicts.append(name)
    if conflicts:
        raise FreezeConflict(
            f"{len(conflicts)} frozen file(s) would change (e.g. {conflicts[:3]}). Instances are "
            "never overwritten: a changed instance needs a new id (suffix v2).")
    cp = checksums_path(root)
    ck = checksums_text(files, fz.sources).encode()
    stale = []
    if replace_set:
        listed = set(parse_checksums(cp)[0]) if cp.exists() else set()
        on_disk = {p.name for pat in ("inst_*.npz", "rulers_*.npz") for p in d.glob(pat)}
        stale = sorted(n for n in (listed | on_disk) - set(files) - tables
                       if n.startswith(("inst_", "rulers_")) and n.endswith(".npz"))
    elif cp.exists() and cp.read_bytes() != ck:
        raise FreezeConflict("CHECKSUMS would change: the frozen set differs from this build "
                             "(use replace_set only for a deliberate, logged change of the set)")
    written = skipped = replaced = 0
    for name, data in files.items():
        p = d / name
        if p.exists():
            if p.read_bytes() == data:
                skipped += 1
                continue
            replaced += 1          # only the two tables can get here (replace_set)
        else:
            written += 1
        atomic_write_bytes(p, data)
    for name in stale:
        (d / name).unlink(missing_ok=True)
    if not cp.exists() or cp.read_bytes() != ck:
        atomic_write_bytes(cp, ck)
    if log:
        log(f"wrote {written} new file(s), {skipped} already frozen and identical, "
            f"{replaced} table(s) replaced, {len(stale)} dropped file(s) removed; "
            f"{len(fz.instances)} instances, {len(fz.seed_table)} scanned draws -> {d}")
    return {"written": written, "skipped": skipped, "replaced": replaced, "removed": len(stale),
            "removed_files": stale}


def freeze(root=None, dataset_dir=None, log=print, replace_set: bool = False, **kw) -> Frozen:
    fz = build(dataset_dir=dataset_dir, log=log, **kw)
    write(fz, root, log=log, replace_set=replace_set)
    return fz


def parse_checksums(path) -> tuple[dict, dict]:
    files, sources = {}, {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("# source "):
            _, _, rel, _, digest = line.split()
            sources[rel.split("/", 1)[1]] = digest
        elif line.strip() and not line.startswith("#"):
            digest, name = line.split(None, 1)
            files[name.strip()] = digest
    return files, sources


def verify(root=None, deep: bool = False, dataset_dir=None, log=print) -> bool:
    d = instances_dir(root)
    cp = checksums_path(root)
    if not cp.exists():
        log(f"FAIL: {cp} does not exist (run `gsp instances freeze`)")
        return False
    files, sources = parse_checksums(cp)
    ok = True
    for name, digest in sorted(files.items()):
        p = d / name
        if not p.exists():
            log(f"FAIL: missing {name}")
            ok = False
        elif sha256_file(p) != digest:
            log(f"FAIL: checksum mismatch {name}")
            ok = False
    listed = set(files)
    extra = sorted(p.name for p in d.iterdir()
                   if p.is_file() and p.name != cp.name and not p.name.startswith(".")
                   and p.name not in listed)
    if extra:
        log(f"FAIL: {len(extra)} unlisted file(s) in {d}: {extra[:5]}")
        ok = False
    dd = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    for name, digest in sorted(sources.items()):
        p = dd / name
        if not p.exists():
            log(f"WARN: dataset source {p} not found (cannot re-derive)")
        elif sha256_file(p) != digest:
            log(f"FAIL: dataset source {name} changed since the freeze")
            ok = False
    n_inst = sum(1 for n in files if n.startswith("inst_"))
    n_rul = sum(1 for n in files if n.startswith("rulers_"))
    log(f"checksums: {len(files)} files ({n_inst} instances, {n_rul} rulers) -> {'OK' if ok else 'FAILED'}")
    if deep and ok:
        ok = verify_deep(root, dataset_dir=dataset_dir, log=log) and ok
    return ok


def verify_deep(root=None, dataset_dir=None, log=print) -> bool:
    """Rebuild everything in memory and compare arrays / tables exactly."""
    d = instances_dir(root)
    table = pd.read_parquet(instances_parquet_path(root))
    Ns = tuple(sorted(table["N"].unique()))
    per_n = int(table.groupby("N")["draw_id"].nunique().iloc[0])
    qs = tuple(sorted(table["q"].unique()))
    fz = build(N_values=Ns, draws_per_n=per_n, q_values=qs, dataset_dir=dataset_dir, log=None)
    ok = True
    for name, data in sorted(fz.files.items()):
        a = load_npz(io.BytesIO(data))
        b = load_npz(d / name)
        if a.keys() != b.keys() or any(
                a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")
                for k in a):
            log(f"FAIL: rebuilt {name} differs")
            ok = False
    seed_now = pd.read_csv(io.BytesIO(seed_table_bytes(fz.seed_table)), float_precision="round_trip")
    seed_disk = pd.read_csv(seed_table_path(root), float_precision="round_trip")
    if not seed_now.equals(seed_disk):
        log("FAIL: rebuilt seed_table differs")
        ok = False
    if not fz.instances.equals(table):
        log("FAIL: rebuilt instances.parquet differs")
        ok = False
    log(f"deep: rebuilt {len(fz.files)} npz files and both tables -> {'OK' if ok else 'FAILED'}")
    return ok


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------

def _med(x) -> str:
    x = np.asarray(x, dtype=float)
    return "—" if x.size == 0 else f"{np.median(x):.4g}"


def table_markdown(root=None) -> str:
    seed = pd.read_csv(seed_table_path(root), float_precision="round_trip")
    inst = pd.read_parquet(instances_parquet_path(root))
    acc = seed[seed["accepted"]]
    L = []
    L.append("# Frozen instances (S1, acceptance revised in S1b)")
    L.append("")
    L.append(f"Generated by `gsp instances table` from `results/instances/` "
             f"(harness_version {HARNESS_VERSION}). Recipe: PLAN §1.1. "
             f"{len(inst)} instances = {inst['draw_id'].nunique()} accepted draws × "
             f"{inst['q'].nunique()} risk weights q ∈ {{{', '.join(repr(float(q)) for q in sorted(inst['q'].unique()))}}}; "
             f"ε = {EPS}, two qubits per asset (n = 2N).")
    L.append("")
    L.append("## Draw acceptance per N")
    L.append("")
    L.append("A draw is accepted iff |F_ε| ≥ 12 at every N (D-14 as revised in S1b; the S1 rule required 24 at "
             "N = 5, 6 and rejected 133 of 163 draws at N = 5). Seed indices e = 0, 1, 2, … are consumed in "
             "order until 30 draws are accepted. Rejected draws stay in `seed_table.csv` with their |F_ε|.")
    L.append("")
    L.append("| N | n | K_req | scanned | accepted | rejected | rejected e (\\|F_ε\\|) |")
    L.append("|---|---|---|---|---|---|---|")
    for N, g in seed.groupby("N"):
        rej = g[~g["accepted"]]
        rej_s = ", ".join(f"{int(r.e)} ({int(r.F_eps)})" for r in rej.itertuples()) or "—"
        kr = int(g["K_req"].iloc[0])
        L.append(f"| {N} | {int(g['n'].iloc[0])} | {kr if kr else '—'} | {len(g)} | {int(g['accepted'].sum())} "
                 f"| {len(rej)} | {rej_s} |")
    L.append("")
    L.append(f"Total rejections: {int((~seed['accepted']).sum())}.")
    L.append("")
    L.append(f"## The K = {K24} subset")
    L.append("")
    L.append(f"The K = {K24} cells (N = {', '.join(str(N) for N in K24_CELL_N)}) run only on the accepted draws "
             f"with |F_ε| ≥ {K24} (column `k24_eligible` in `seed_table.csv` and `instances.parquet`). They are "
             "reported with their draw count and enter neither D1 nor D2's 12-cell grid. Other sizes are "
             "listed for reference only.")
    L.append("")
    L.append(f"| N | K = {K24} cell | eligible draws (of accepted) | eligible e |")
    L.append("|---|---|---|---|")
    for N, g in seed.groupby("N"):
        el = g[g["k24_eligible"].astype(bool)]
        es = ", ".join(str(int(e)) for e in el["e"]) if N in K24_CELL_N else "—"
        L.append(f"| {N} | {'yes' if N in K24_CELL_N else 'no'} | {len(el)}/{int(g['accepted'].sum())} | {es} |")
    L.append("")
    L.append("## |F_ε| per (N, K)")
    L.append("")
    L.append("|F_ε| = number of strings within the ε-band (the band depends only on the draw, so all "
             "three q share it). Columns K: accepted draws (of 30) whose band holds at least K strings, "
             "i.e. can host a K-string sector inside the band; in brackets the same over every scanned draw.")
    L.append("")
    hdr = "| N | n | 2^n | min | median | max | " + " | ".join(f"K = {K}" for K in K_TABLE) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (6 + len(K_TABLE)))
    for N, g in seed.groupby("N"):
        a = g[g["accepted"]]
        n = int(g["n"].iloc[0])
        cells = [f"{int((a['F_eps'] >= K).sum())}/{len(a)} ({int((g['F_eps'] >= K).sum())}/{len(g)})"
                 for K in K_TABLE]
        L.append(f"| {N} | {n} | {1 << n} | {int(a['F_eps'].min())} | {_med(a['F_eps'])} | "
                 f"{int(a['F_eps'].max())} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Rulers per N (medians over accepted draws and q)")
    L.append("")
    L.append("| N | \\|F_ε\\|/2^n | \\|X*\\| (max) | top-10 set size (max) | distinct band values | ladder gap ḡ (normalized) | E_max − E_min | Jh boost of H_obj |")
    L.append("|---|---|---|---|---|---|---|---|")
    for N, g in inst.groupby("N"):
        n = int(g["n"].iloc[0])
        L.append(f"| {N} | {_med(g['F_size'] / (1 << n))} | {_med(g['n_xstar'])} ({int(g['n_xstar'].max())}) "
                 f"| {_med(g['top10_size'])} ({int(g['top10_size'].max())}) | {_med(g['n_distinct'])} "
                 f"| {_med(g['gap_band'])} | {_med(g['E_max'] - g['E_min'])} | {_med(g['boost_obj'])} |")
    L.append("")
    dis = int(seed["n_direct_disagree"].sum())
    near = int(seed["n_near_boundary"].sum())
    L.append("## Band definition check")
    L.append("")
    L.append("The band is evaluated exactly as the completed confined runs did (λ = 1 penalty QUBO through "
             "`all_state_to_return`, |state_penalty| ≤ ε²). Cross-check against the direct |P·x − 1| ≤ ε over "
             f"all 2^n strings of every scanned draw: **{dis}** disagreeing strings; **{near}** strings within "
             "1e-12 of the boundary.")
    L.append("")
    return "\n".join(L)


def write_report(root=None) -> Path:
    p = reports_dir() / "instances.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(table_markdown(root))
    return p
