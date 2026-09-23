"""`gsp instances freeze | verify | table` on a small subset, and checks of the real frozen set."""

import numpy as np
import pandas as pd
import pytest

from gsp.cli import main as cli_main
from gsp.instances import freeze as fz
from gsp.instances.draws import (DRAWS_PER_N, K_REQ, N_VALUES, Q_VALUES, draw_seed, ga_seed,
                                 inst_id, restart_seed)
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
        assert row.accepted == (row.F_eps >= K_REQ.get(row.N, 0))
    # seed indices are consumed in order, rejected ones included
    for N, g in seed.groupby("N"):
        assert list(g["e"]) == list(range(len(g)))
        assert g["accepted"].sum() == 3 and bool(g["accepted"].iloc[-1])


def test_freeze_idempotent_and_never_overwrites(small):
    res = fz.write(fz.build(N_values=(4, 5, 8), draws_per_n=3, log=None), small, log=None)
    assert res == {"written": 0, "skipped": 56}    # 27 inst + 27 rulers + 2 tables
    # a different build for the same ids must be refused, not written
    other = fz.build(N_values=(4, 5, 8), draws_per_n=3, q_values=(1.0, 1.5, 2.0), log=None)
    with pytest.raises(fz.FreezeConflict):
        fz.write(other, small, log=None)
    assert fz.verify(root=small, log=lambda *a: None)


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
    assert (acc["F_eps"] >= acc["N"].map(lambda N: K_REQ.get(N, 0))).all()
    assert "N05e000q1.5" in set(inst["inst_id"])
