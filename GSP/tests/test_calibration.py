"""S9a: the calibration queues (`gsp.runner.calibration`), the hashed evidence flags (O-2 / O-11), D1's exclusion of
evidence runs, the proposed gate-matched depths, and the launcher `scripts/s9_calibration.sh` (CPU only: a fake
python / nvidia-smi stand in for the GPU)."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsp.arms.base import evidence_tag, is_evidence, make_arm
from gsp.instances.instance import load_instance
from gsp.runner import calibration as C
from gsp.store.paths import GSP_ROOT, inst_path

SCRIPTS = GSP_ROOT / "scripts"
I4 = "N04e004q1.5"
RING12 = {"connectivity": "ring", "rule": "violation", "K": 12}

needs_store = pytest.mark.skipif(not inst_path(I4).exists(), reason="the frozen instance set is not built")


# --- the hashed flags ------------------------------------------------------------------------------------------------
@needs_store
def test_default_run_ids_unchanged_by_the_new_flags():
    inst = load_instance(I4)
    a0, a1, a3 = make_arm("A0"), make_arm("A1"), make_arm("A3")
    base0 = a0.config(inst, None, 7, None, lam=0.005).run_id
    assert a0.config(inst, None, 7, None, lam=0.005, circuit_boosted=False, evidence=None).run_id == base0
    b0 = a0.config(inst, None, 7, None, lam=0.005, circuit_boosted=True, evidence="O-2")
    assert b0.run_id != base0 and b0.extra("circuit_boosted") is True and b0.extra("evidence") == "O-2"
    assert a0.config(inst, None, 7, None, lam=0.005, circuit_boosted=True).run_id not in (base0, b0.run_id)
    base1 = a1.config(inst, RING12, 5, None).run_id
    assert a1.config(inst, RING12, 5, None, circuit_boosted=False).run_id == base1
    assert a1.config(inst, RING12, 5, None, circuit_boosted=True, evidence="O-2").run_id != base1
    base3 = a3.config(inst, None, 5, None, lam=0.005).run_id
    assert a3.config(inst, None, 5, None, lam=0.005, tikhonov=1e-6, psd_project=False).run_id == base3
    ids = {base3,
           a3.config(inst, None, 5, None, lam=0.005, tikhonov=1e-4, evidence="O-11").run_id,
           a3.config(inst, None, 5, None, lam=0.005, psd_project=True, evidence="O-11").run_id,
           a3.config(inst, None, 5, None, lam=0.005, theta0_jitter=1e-9, theta0_jitter_seed=0, evidence="O-11").run_id,
           a3.config(inst, None, 5, None, lam=0.005, theta0_jitter=1e-9, theta0_jitter_seed=1, evidence="O-11").run_id,
           a3.config(inst, None, 5, None, lam=0.005, theta0_jitter=1e-9, theta0_jitter_seed=0).run_id}
    assert len(ids) == 6
    with pytest.raises(ValueError):
        a3.config(inst, None, 5, None, lam=0.005, theta0_jitter=1e-9)
    with pytest.raises(ValueError):
        a3.config(inst, None, 5, None, lam=0.005, tikhonov=0.0)


@needs_store
def test_stored_s7_records_still_hash_the_same():
    """Every stored A3 / A3d / A0 / A1 record rebuilds its run_id through the arm's config with the new signature."""
    from gsp.store.index import load_registry
    reg = load_registry()
    if reg.empty:
        pytest.skip("no stored runs")
    inst_cache = {}
    n = 0
    for r in reg[reg["arm"].isin(["A3", "A3d"])].itertuples():
        inst = inst_cache.setdefault(r.inst_id, load_instance(r.inst_id))
        cfg = make_arm(r.arm).config(inst, None, int(r.effort), None, lam=float(r.lam), n_steps=int(r.n_steps))
        assert cfg.run_id == r.run_id
        n += 1
    assert n >= 1


def test_evidence_tag_validation():
    assert evidence_tag(None) is None and evidence_tag("O-11") == "O-11"
    for bad in ("", " O-2", "O-2 "):
        with pytest.raises(ValueError):
            evidence_tag(bad)
    assert is_evidence({"evidence": "O-2"}) and not is_evidence({"evidence": float("nan")}) and not is_evidence({})


# --- O-11 variant (b): PSD projection --------------------------------------------------------------------------------
def test_psd_project():
    from gsp.train.mclachlan import psd_project
    rng = np.random.default_rng(3)
    V = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    w = np.array([-4e-3, -1e-5, 1e-6, 0.2, 1.0])
    A = (V * w) @ V.T
    P, clip_sum, clip_n = psd_project(A)
    assert clip_n == 2 and abs(clip_sum - (4e-3 + 1e-5)) < 1e-15
    wp = np.linalg.eigvalsh(P)
    assert wp.min() > -1e-15 and np.allclose(np.sort(wp)[2:], [1e-6, 0.2, 1.0], atol=1e-14)
    Q, s, k = psd_project(np.diag([1.0, 2.0]))
    assert k == 0 and s == 0.0 and np.array_equal(Q, np.diag([1.0, 2.0]))


def test_psd_flag_in_the_loop_only_when_set():
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.instances.encode import Ising
    from gsp.train.mclachlan import McLachlanConfig, ramp_init, run_mclachlan
    from tests.helpers.np_varqite import NumpyEngine
    rng = np.random.default_rng(0)
    n = 3
    H = Ising(n=n, const=0.0, h=rng.normal(size=n), J=np.triu(rng.normal(size=(n, n)), 1),
              has_h=np.ones(n, bool), has_J=np.triu(np.ones((n, n), bool), 1))
    A = penalty_ansatz(H, 2, alpha=1.0)
    x0 = ramp_init(2)
    plain = run_mclachlan(NumpyEngine(A), x0, McLachlanConfig(metric="M1", n_steps=4, f_tol=-1.0))
    proj = run_mclachlan(NumpyEngine(A), x0, McLachlanConfig(metric="M1", n_steps=4, f_tol=-1.0, psd=True))
    assert "psd_clip_sum" not in plain.step and "psd_clip_sum" in proj.step
    assert proj.step["psd_clip_sum"].shape == (5,) and np.all(proj.step["psd_clip_sum"][1:] >= 0)
    if not np.any(proj.step["psd_clip_n"][1:] > 0):          # nothing clipped: the two loops are identical
        assert np.array_equal(plain.E_loop, proj.E_loop)


# --- D1 never reads an evidence run ----------------------------------------------------------------------------------
def test_d1_match_drops_evidence_runs():
    from gsp.stats import d1
    reg = pd.DataFrame([
        {"arm": "A0", "status": "done", "init": "random", "lam": 0.005, "N": 5, "run_id": "a", "evidence": None},
        {"arm": "A0", "status": "done", "init": "random", "lam": 0.005, "N": 5, "run_id": "b", "evidence": "O-2"},
        {"arm": "A3", "status": "done", "lam": 0.005, "N": 7, "metric": "M1", "n_steps": 300, "run_id": "c",
         "evidence": "O-11"},
        {"arm": "A3", "status": "done", "lam": 0.005, "N": 7, "metric": "M1", "n_steps": 300, "run_id": "d",
         "evidence": np.nan}])
    spec = d1.d1_spec(lam_star={5: 0.005, 7: 0.005})
    assert list(d1._match(reg, "A0", spec["A0"])["run_id"]) == ["a"]
    assert list(d1._match(reg, "A3", spec["A3"])["run_id"]) == ["d"]
    assert list(d1._match(reg.drop(columns="evidence"), "A0", spec["A0"])["run_id"]) == ["a", "b"]


# --- A. gate-matched depths ------------------------------------------------------------------------------------------
def test_matched_depths_rule_on_synthetic_rows():
    rows = []
    for d, cx in enumerate([1000, 1100, 1210, 1300]):          # median 1155 -> ceil(1155 / 56) = 21
        for q in (1.0, 1.5, 3.0):
            rows.append({"N": 4, "n": 8, "draw_id": f"d{d}", "inst_id": f"d{d}q{q}", "q": q, "a0_layer_cx": 56,
                         "a0_layer_cx_distinct": 1, "a1_L5_cx_ii": cx, "a1_L5_cx_ii_S": cx // 3,
                         "a1_L5_cx_ii_rank": cx + 10})
    det = C.matched_depths(pd.DataFrame(rows), L1s=(5,))
    d = det[4][5]
    assert d["L0"] == 21 and d["a1_cx_ii_median"] == 1155.0 and d["draws"] == 4
    assert d["L0_per_draw_min"] == 18 and d["L0_per_draw_max"] == 24 and d["info_L0_ring_rank"] == 21
    bad = pd.DataFrame(rows)
    bad.loc[0, "a1_L5_cx_ii"] += 2                              # the q of a draw disagree
    with pytest.raises(ValueError):
        C.matched_depths(bad, L1s=(5,))


@needs_store
def test_matched_depths_proposed_values():
    """PLAN §1.3 on the frozen set (S9a's proposal; S9b finalizes): pinned, and equal to S3's counts."""
    rows = C.matched_depth_rows()
    det = C.matched_depths(rows)
    assert {N: {L1: d["L0"] for L1, d in m.items()} for N, m in det.items()} == {
        4: {5: 59, 7: 79, 9: 99}, 5: {5: 48, 7: 65, 9: 81}, 6: {5: 41, 7: 55, 9: 69}, 7: {5: 36, 7: 48, 9: 61}}
    assert all(d["a0_layer_cx"] == d["a0_layer_cx_formula"] for m in det.values() for d in m.values())
    chk = C.check_against_s3(rows)
    if chk.get("available"):
        assert chk["missing_in_s3"] == 0 and max(v for k, v in chk.items() if k.startswith("max_abs")) == 0.0


# --- B-D. the queues -------------------------------------------------------------------------------------------------
@needs_store
def test_timing_and_evidence_queues():
    L0 = {5: {9: 81}, 7: {9: 61}}
    q1 = C.resolve_strict(C.timing_specs(L0))
    assert len(q1) == 24 and len({s["run_id"] for s in q1}) == 24
    arms = pd.Series([s["arm"] for s in q1]).value_counts().to_dict()
    assert arms == {"A0": 4, "A1": 6, "A2p": 4, "A2c": 4, "A3": 3, "A3d": 1, "A4": 1, "A6": 1}
    assert {s["effort"] for s in q1 if s["arm"] == "A0"} == {9, 81, 61}
    assert all(s["kw"].get("n_steps") == 20 for s in q1 if s["arm"] in ("A3", "A3d"))
    assert all(s["effort"] == 10 and s["kw"]["step_units"] == "normalized" for s in q1 if s["arm"] in ("A4", "A6"))
    assert not any("evidence" in s["kw"] for s in q1)
    k24 = [s for s in q1 if (s["cell"] or {}).get("K") == 24]
    assert [s["inst_id"] for s in k24] == C.first_draws(5, k24=True) + C.first_draws(6, k24=True)
    q3 = C.resolve_strict(C.evidence_specs())
    assert len(q3) == 125 and len({s["run_id"] for s in q3}) == 125
    ev = [s for s in q3 if "evidence" in s["kw"]]
    assert len(ev) == 40 + 20 + 40
    assert all(s["kw"]["evidence"] == "O-2" and s["kw"]["circuit_boosted"] for s in ev if s["arm"] in ("A0", "A1"))
    plain = [s for s in q3 if "evidence" not in s["kw"]]
    assert len(plain) == 25 and all(not s["kw"].get("circuit_boosted") for s in plain)
    assert {s["arm"] for s in plain} == {"A1", "A3"}
    a3_plain = [s for s in plain if s["arm"] == "A3"]
    assert all(set(s["kw"]) == {"lam"} for s in a3_plain)             # O-11 (a) without jitter = the sweep config
    o2_draws = {s["inst_id"] for s in q3 if s["arm"] == "A0" and s["N"] == 7}
    assert o2_draws == set(C.first_draws(7, 10))
    # the pilot counterparts of O-2's boosted A0 runs exist in queue 2 (same draw, lam, depth; un-boosted)
    q2, _ = C.pilot_queue_specs()
    pil = {(s["inst_id"], s["lam"], s["effort"]) for s in q2}
    assert all((s["inst_id"], s["kw"]["lam"], s["effort"]) in pil for s in q3 if s["arm"] == "A0")
    # strict kwargs: a flag the arm does not accept is refused, never dropped
    with pytest.raises(Exception):
        C.resolve_strict([C.spec("A2p", I4, 5, kw={"lam": 0.005, "evidence": "O-2"}, tag="t", label="bad")])


def test_estimate_spec_rates():
    s = C.spec("A3", "N07e000q1.5", 9, kw={"lam": 0.005, "n_steps": 20}, tag="timing", label="x")
    e = C.estimate_spec(s)
    assert e["source"] == "measured (S7)" and abs(e["est_s"] - (20 * (1.903 + C.logger_s(14)) + 0.5 + 2.0)) < 1e-9
    s = C.spec("A0", "N10e000q1.5", 9, kw={"lam": 0.005}, tag="timing", label="x")
    assert C.estimate_spec(s)["s_per_unit"] == C.RATES[("A0", 20, 9)]
    s = C.spec("A0", "N07e000q1.5", 61, kw={"lam": 0.005}, tag="timing", label="x")
    assert C.estimate_spec(s)["est_s"] is None
    assert C.estimate_spec(s, {C.bench_key(s): 0.04})["s_per_unit"] == pytest.approx(123 * 0.04)


# --- E. the launcher -------------------------------------------------------------------------------------------------
FAKE_PY = r"""#!/usr/bin/env bash
# fake python for the launcher test: `-m gsp.cli run --queue Q ...` exits FAKE_RC_<queue stem> (default 0)
echo "fakepy $*" >> "$FAKE_LOG"
if [ "$3" = "run" ]; then
  q=$(basename "$5" .jsonl); v="FAKE_RC_$q"; exit "${!v:-0}"
fi
exit 0
"""


def _launcher_env(tmp_path, **extra):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n")
    (b / "nvidia-smi").chmod(0o755)
    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    res = tmp_path / "res"
    qs = []
    for name in ("qa", "qb"):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("")
        qs.append(str(p))
    env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", GSP_PY=str(py), GSP_RESULTS=str(res),
               S9_QUEUES=" ".join(qs), FAKE_LOG=str(tmp_path / "fake.log"), S9_GPU_WAIT_S="0")
    env.update(extra)
    return env, res


def _run_launcher(env):
    return subprocess.run(["bash", str(SCRIPTS / "s9_calibration.sh")], env=env, capture_output=True, text=True,
                          timeout=120)


def _calls(tmp_path):
    p = tmp_path / "fake.log"
    return p.read_text().splitlines() if p.exists() else []


def test_launcher_sequence_and_exit_codes(tmp_path):
    env, res = _launcher_env(tmp_path)
    r = _run_launcher(env)
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""               # prints nothing
    calls = _calls(tmp_path)
    runs = [c for c in calls if " run --queue " in c]
    assert [Path(c.split("--queue ")[1].split()[0]).stem for c in runs] == ["qa", "qb"]
    assert any("gsp.cli index" in c for c in calls) and any("gsp.cli aggregate" in c for c in calls)
    log = (res / "logs" / "s9_calibration.log").read_text()
    assert "done (any failed runs: 0)" in log
    # failed runs in a queue: continue, exit 1 at the end
    (tmp_path / "fake.log").unlink()
    r = _run_launcher(dict(env, FAKE_RC_qa="1"))
    assert r.returncode == 1 and len([c for c in _calls(tmp_path) if " run --queue " in c]) == 2
    # a guard failure in the first queue: stop, the second queue never starts, no index / aggregate
    (tmp_path / "fake.log").unlink()
    r = _run_launcher(dict(env, FAKE_RC_qa="3"))
    assert r.returncode == 3
    assert [c for c in _calls(tmp_path) if " run --queue " in c][-1].count("qa.jsonl") == 1
    assert not any("qb.jsonl" in c or "index" in c for c in _calls(tmp_path))
    # a signal-stopped queue: stop with its code
    (tmp_path / "fake.log").unlink()
    assert _run_launcher(dict(env, FAKE_RC_qa="143")).returncode == 143
    assert not any("qb.jsonl" in c for c in _calls(tmp_path))
    # usage: a missing queue file
    assert _run_launcher(dict(env, S9_QUEUES=str(tmp_path / "nope.jsonl"))).returncode == 2


def test_launcher_guards(tmp_path):
    env, res = _launcher_env(tmp_path)
    (res / "logs").mkdir(parents=True, exist_ok=True)
    fd = os.open(res / "logs" / "s9_calibration.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert _run_launcher(env).returncode == 3                                 # another launcher
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert not _calls(tmp_path)
    (tmp_path / "bin" / "nvidia-smi").write_text('#!/usr/bin/env bash\necho "4242, python"\n')
    assert _run_launcher(env).returncode == 3                                     # the GPU stays busy
    (tmp_path / "bin" / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 9\n")
    assert _run_launcher(env).returncode == 3                                     # nvidia-smi fails
    assert not _calls(tmp_path)
    assert "guard" in (res / "logs" / "s9_calibration.log").read_text()
