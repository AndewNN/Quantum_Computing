"""`gsp instances freeze | verify | table` on a small subset, and checks of the real frozen set."""

import numpy as np
import pandas as pd
import pytest

from gsp.cli import main as cli_main
from gsp.instances import freeze as fz
from gsp.instances.draws import (DRAWS_PER_N, K24, K_REQ, N_VALUES, Q_VALUES, draw_seed, ga_seed,
                                 inst_id, k_req, restart_seed)
from gsp.instances.instance import load_instance, load_rulers, load_seed_table
from gsp.store import paths


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("res")
    fz.freeze(root=root, N_values=(4, 5, 8), draws_per_n=3, log=None)
    return root


def test_small_freeze_verify(small):
    assert fz.verify(root=small, log=lambda *a: None)
    assert fz.verify(root=small, deep=True, log=lambda *a: None)
    inst = pd.read_parquet(paths.instances_parquet_path(small))
    assert len(inst) == 3 * 3 * 3
    assert list(inst.groupby("N").size()) == [9, 9, 9]
    seed = load_seed_table(small)
    for row in seed.itertuples():
        assert row.draw_seed == draw_seed(row.N, row.e)
        assert row.restart_seed_r4 == restart_seed(row.N, row.e, 4)
        assert row.ga_seed_objective == ga_seed(row.N, row.e, 1)
        assert K_REQ == k_req(row.N) == row.K_req == 12
        assert row.accepted == (row.F_eps >= 12)
        assert row.k24_eligible == (row.accepted and row.F_eps >= K24)
    # seed indices are consumed in order, rejected ones included
    for N, g in seed.groupby("N"):
        assert list(g["e"]) == list(range(len(g)))
        assert g["accepted"].sum() == 3 and bool(g["accepted"].iloc[-1])


def test_freeze_idempotent_and_never_overwrites(small):
    res = fz.write(fz.build(N_values=(4, 5, 8), draws_per_n=3, log=None), small, log=None)
    assert (res["written"], res["skipped"], res["replaced"], res["removed"]) == (0, 56, 0, 0)  # 27+27+2 tables
    # a different build for the same ids must be refused, not written
    other = fz.build(N_values=(4, 5, 8), draws_per_n=3, q_values=(1.0, 1.5, 2.0), log=None)
    with pytest.raises(fz.FreezeConflict):
        fz.write(other, small, log=None)
    assert fz.verify(root=small, log=lambda *a: None)


def test_replace_set_keeps_survivors_and_drops_the_rest(tmp_path):
    fz.freeze(root=tmp_path, N_values=(4, 5), draws_per_n=3, log=None)
    d = paths.instances_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in d.glob("*.npz")}
    # strict mode refuses a different set; nothing is touched
    with pytest.raises(fz.FreezeConflict):
        fz.freeze(root=tmp_path, N_values=(4, 5), draws_per_n=2, log=None)
    assert {p.name: p.read_bytes() for p in d.glob("*.npz")} == before
    # replace_set: survivors byte-identical, the dropped draws' files removed, tables regenerated
    res = fz.write(fz.build(N_values=(4, 5), draws_per_n=2, log=None), tmp_path, log=None, replace_set=True)
    after = {p.name: p.read_bytes() for p in d.glob("*.npz")}
    assert res["removed"] == 12 and res["written"] == 0 and len(after) == 24
    assert all(before[k] == v for k, v in after.items())
    assert set(res["removed_files"]) == set(before) - set(after)
    assert fz.verify(root=tmp_path, deep=True, log=lambda *a: None)
    # a surviving id whose bytes would change is still refused under replace_set
    victim = sorted(d.glob("inst_*.npz"))[0]
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    with pytest.raises(fz.FreezeConflict):
        fz.write(fz.build(N_values=(4, 5), draws_per_n=2, log=None), tmp_path, log=None, replace_set=True)


def test_verify_detects_tampering(tmp_path):
    fz.freeze(root=tmp_path, N_values=(4,), draws_per_n=1, log=None)
    victim = sorted(paths.instances_dir(tmp_path).glob("rulers_*.npz"))[0]
    data = bytearray(victim.read_bytes())
    data[-40] ^= 0xFF
    victim.write_bytes(bytes(data))
    msgs = []
    assert not fz.verify(root=tmp_path, log=msgs.append)
    assert any("checksum mismatch" in m for m in msgs)


def test_instance_loader_roundtrip(small):
    table = pd.read_parquet(paths.instances_parquet_path(small))
    iid = table["inst_id"].iloc[0]
    inst = load_instance(iid, small)
    rul = load_rulers(iid, small)
    assert inst.inst_id == iid and inst.n == 2 * inst.N
    assert rul.F_size == table["F_size"].iloc[0] and rul.E_min == table["E_min"].iloc[0]
    # H(lam) diagonal (Ising form) == QUBO form to rounding
    from gsp.instances.encode import qubo_energies
    H = inst.hamiltonian(0.005)
    idx = np.arange(1 << inst.n)
    a = H.diagonal(idx)
    b = -qubo_energies(inst.qubo(0.005), 0.005)
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b))
    # rulers' f is H_obj on the band
    assert np.max(np.abs(inst.H_obj.diagonal(rul.band_idx) - rul.f_band)) <= 1e-12 * np.max(np.abs(rul.f_band))


def test_cli_table(small, capsys):
    assert cli_main(["instances", "table", "--no-write", "--results", str(small)]) == 0
    out = capsys.readouterr().out
    assert "| 4 | 8 | 12 |" in out and "K = 24" in out


# ------------------------------------------------------------------------------------------
# the real frozen set (skipped until `gsp instances freeze` has run)
# ------------------------------------------------------------------------------------------

real = pytest.mark.skipif(not paths.checksums_path().exists(), reason="instances not frozen yet")


@real
def test_real_freeze_complete_and_verified():
    assert fz.verify(log=lambda *a: None)
    inst = pd.read_parquet(paths.instances_parquet_path())
    assert len(inst) == len(N_VALUES) * DRAWS_PER_N * len(Q_VALUES) == 630
    assert set(inst.groupby("N")["draw_id"].nunique()) == {DRAWS_PER_N}
    assert set(inst["q"]) == set(Q_VALUES)
    assert inst["inst_id"].is_unique
    assert all(inst["inst_id"] == [inst_id(N, e, q) for N, e, q in zip(inst["N"], inst["e"], inst["q"])])
    seed = load_seed_table()
    acc = seed[seed["accepted"]]
    assert len(acc) == 210
    assert (acc["F_eps"] >= 12).all() and (seed.loc[~seed["accepted"], "F_eps"] < 12).all()
    assert "N05e000q1.5" in set(inst["inst_id"]) and "N05e001q1.5" in set(inst["inst_id"])
    # S1b numbers (uniform |F_eps| >= 12): rejections per N and the K = 24 subsets
    rej = seed[~seed["accepted"]].groupby("N").size().reindex(N_VALUES, fill_value=0)
    assert list(rej) == [83, 3, 2, 1, 0, 0, 0]
    k24 = acc[acc["k24_eligible"]].groupby("N").size()
    assert (k24.get(5), k24.get(6)) == (10, 21)
    assert list(acc.loc[(acc["N"] == 5) & acc["k24_eligible"], "e"]) == [0, 7, 8, 15, 17, 19, 22, 24, 29, 31]
    assert (inst.groupby("draw_id")["k24_eligible"].nunique() == 1).all()
    assert inst["k24_eligible"].sum() == 3 * int(acc["k24_eligible"].sum())
