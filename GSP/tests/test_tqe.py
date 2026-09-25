"""TQE rerun (reports/tqe_rerun_plan.md): the A0 / A1 flags (hashed only when set), the Exp1-4 planner, the
dynamic queue runner and `gsp q` edits. CPU only (the queue tests use the fake arm)."""

import json
import threading
import time

import numpy as np
import pytest

from gsp.arms.qaoa import A0, A1
from gsp.instances.adhoc import adhoc_instance
from gsp.runner import qedit, tqe
from gsp.runner.queue import Runner, run_state
from gsp.train.schedules import ramp_params
from tests.helpers.fake_arm import FAKE, factory, fake_spec, install_fakes, write_specs

RING12 = {"connectivity": "ring", "rule": "violation", "K": 12}


@pytest.fixture(scope="module")
def inst5():
    return adhoc_instance(5, 0, 1.5)[0]


def test_flags_hash_only_when_set(inst5):
    a0, a1 = A0(), A1()
    b0 = a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True)
    assert a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, ramp=None, alpha=None, weight_decay=None).run_id \
        == b0.run_id
    ids = {b0.run_id,
           a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, alpha=5000.0).run_id,
           a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, weight_decay=0.0).run_id,
           a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, init="ramp", ramp=(1.5, 3.0)).run_id,
           a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, init="ramp", ramp=(0.4, 2.5)).run_id}
    assert len(ids) == 5
    assert "alpha_override" not in dict(b0.extras) and "weight_decay" not in dict(b0.extras)
    c = a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, init="ramp", ramp=(1.5, 3.0), weight_decay=0.0)
    assert dict(c.extras)["ramp_dbeta"] == 1.5 and dict(c.extras)["weight_decay"] == 0.0
    with pytest.raises(ValueError):
        a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, init="ramp")
    with pytest.raises(ValueError):
        a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, ramp=(1.5, 3.0))
    with pytest.raises(ValueError):
        a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, alpha=0.0)
    kw = dict(adhoc=True, sector_source="bf", circuit_boosted=True)
    b1 = a1.config(inst5, RING12, 5, 123, **kw)
    assert a1.config(inst5, RING12, 5, 123, alpha=None, **kw).run_id == b1.run_id
    assert a1.config(inst5, RING12, 5, 123, alpha=10500.0, **kw).run_id != b1.run_id


def test_execute_passes_ramp_alpha_and_weight_decay(inst5, monkeypatch):
    """execute() hands train_adamw the ramp in boosted units (RAMP_SIGN) and the weight decay; the ansatz carries the
    override alpha. train_adamw is stubbed, so nothing is simulated."""
    import gsp.arms.qaoa as Q

    class Stop(Exception):
        pass

    got = {}

    def fake_train(energy, x0, opt, grad, logger=None):
        got.update(x0=np.array(x0), wd=opt.weight_decay, alpha=energy.__self__.alpha)
        raise Stop

    monkeypatch.setattr(Q, "train_adamw", fake_train)
    a0 = A0()
    cfg = a0.config(inst5, None, 7, 123, lam=0.005, adhoc=True, circuit_boosted=True, init="ramp",
                    ramp=(1.5, 3.0), weight_decay=0.0)
    with pytest.raises(Stop):
        a0.execute(cfg, inst5, adhoc_instance(5, 0, 1.5)[1], None, logger=False)
    assert np.allclose(got["x0"], ramp_params(7, 1.5, 3.0, 1.0)) and got["wd"] == 0.0
    assert got["x0"][7] < 0 and got["x0"][0] > 0                 # beta negated (RAMP_SIGN), gamma > 0
    cfg = a0.config(inst5, None, 5, 123, lam=0.005, adhoc=True, circuit_boosted=True, alpha=5000.0)
    with pytest.raises(Stop):
        a0.execute(cfg, inst5, adhoc_instance(5, 0, 1.5)[1], None, logger=False)
    assert got["alpha"] == 5000.0 and got["wd"] == 0.01
    a1 = A1()
    cfg = a1.config(inst5, RING12, 5, 123, adhoc=True, sector_source="bf", circuit_boosted=True, alpha=10500.0)
    with pytest.raises(Stop):
        a1.execute(cfg, inst5, adhoc_instance(5, 0, 1.5)[1], RING12, logger=False)
    assert got["alpha"] == 10500.0


def test_tqe_plan_counts_and_seeds():
    S = tqe.all_specs()
    assert len(S) == 6600
    by = {}
    for s in S:
        by[(s["method"], s.get("K"))] = by.get((s["method"], s.get("K")), 0) + 1
    assert by == {("SP", None): 4500, ("SC", 12): 900, ("SC", 24): 900, ("SC", 48): 300}
    assert not [s for s in S if s["exp"] == 1 and s["N"] >= 9]
    assert {s["kw"]["alpha"] for s in S if s["exp"] == 1 and s["method"] == "SC" and s["N"] == 8} == {1300.0}
    assert {s["kw"]["alpha"] for s in S if s["exp"] == 1 and s["method"] == "SP" and s["N"] == 8
            and s["lam"] == 5.0} == {2.5}
    assert all("alpha" not in s["kw"] for s in S if s["exp"] != 1)
    assert all(s["kw"]["ramp"] == [1.5, 3.0] and s["kw"]["init"] == "ramp" for s in S if s["exp"] in (3, 4))
    assert all(s["kw"].get("weight_decay") == 0.0 for s in S if s["exp"] == 4)
    assert all(s["kw"]["circuit_boosted"] for s in S)
    assert tqe.restart_seed_formula(4, 0) == 23997                  # = seed table N04e000 restart_seed_r0
    parts = [s["tier"] for s in S]
    assert parts.index("sc_k24") > parts.index("sc_k12_ext") > max(i for i, p in enumerate(parts) if p == "paper")
    specs, hours = tqe.split(tqe.all_specs(exps=(1, 2)), {"a": 1.0, "b": 0.5})
    assert len(specs["a"]) + len(specs["b"]) == len(tqe.all_specs(exps=(1, 2)))
    assert abs(hours["a"] - hours["b"]) < 0.2                        # greedy: within one long run


def test_tqe_resolve_unique_ids(tmp_path):
    R = tqe.resolve_specs(tqe.part_specs("paper", (1, 2, 3, 4))[:400], tmp_path)
    assert len({r["run_id"] for r in R}) == 400
    assert all(dict(r["kw"]) for r in R)
    (tmp_path / "instances").mkdir()
    (tmp_path / "instances" / "inst_N05e000q1.5.npz").write_bytes(b"")
    with pytest.raises(ValueError):
        tqe.resolve_specs(R[:1], tmp_path)


@pytest.fixture
def fakes(monkeypatch):
    install_fakes(monkeypatch.setattr)


def test_dynamic_runner_follows_live_edits(tmp_path, fakes):
    specs = [fake_spec(e) for e in (1, 2, 3, 4)]
    new = fake_spec(5)
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    order = []

    def probe(cfg):
        order.append(cfg.effort)

    def editor():
        t_end = time.time() + 20
        while time.time() < t_end and run_state(FAKE, specs[0]["run_id"], out)[0] != "done":
            time.sleep(0.02)
        src = write_specs(tmp_path / "new.jsonl", [new])
        assert qedit.rm(q, specs[2]["run_id"]) == 1
        assert qedit.add(q, src, top=True) == 1

    th = threading.Thread(target=editor)
    th.start()
    r = Runner(q, None, out, gpu=False, aggregate=False, echo=False, heartbeat_s=0.05, dynamic=True,
               arm_factory=factory(steps=10, dt=0.05, probe=probe)).run()
    th.join()
    assert r["exit_code"] == 0 and r["state"] == "finished"
    assert run_state(FAKE, specs[2]["run_id"], out)[0] == "absent"        # removed before it started
    assert order[0] == 1 and 3 not in order and set(order) == {1, 2, 4, 5}
    assert order.index(5) < order.index(4)                                 # the prepended line ran first


def test_q_take_and_ls(tmp_path, fakes):
    specs = [fake_spec(e) for e in (1, 2, 3, 4)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    Runner(q, None, out, gpu=False, aggregate=False, echo=False, heartbeat_s=0.05, max_runs=1,
           arm_factory=factory()).run()
    info = qedit.ls(q, out)
    assert info["lines"] == 4 and info["pending"] == 3 and info["states"]["done"] == 1
    assert qedit.take(q, tmp_path / "b.jsonl", tail=2, out_root=out) == 2
    assert [json.loads(x)["effort"] for x in (tmp_path / "b.jsonl").read_text().splitlines()] == [3, 4]
    assert [json.loads(x)["effort"] for x in q.read_text().splitlines()] == [1, 2]
    assert qedit.take(q, tmp_path / "b.jsonl", match='"effort":1', out_root=out) == 0     # done lines stay
    assert qedit.top(q, '"effort": 2') == 1
