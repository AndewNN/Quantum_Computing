"""S6: `gsp plan` (the OFAT set of PLAN §1.2), the queue runner, `gsp missing`, `gsp merge`, scripts/queue.sh and
scripts/remote/. CPU only: the runner is driven by a fake arm (tests/helpers/fake_arm.py) with the real signal
handling, including a subprocess killed with SIGTERM / SIGINT / SIGKILL."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gsp.runner import merge as MG
from gsp.runner import missing as M
from gsp.runner import plan as P
from gsp.runner.queue import QueueBusy, Runner, lock_path, postrun_state, progress_path, run_state
from gsp.store.paths import GSP_ROOT, run_dir, run_log_path
from tests.helpers.fake_arm import FAKE, FakeArm, factory, fake_spec, install_fakes, write_specs

ENV = P.load_envelope()
SCRIPTS = GSP_ROOT / "scripts"


# --- plan --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def full_grid():
    return P.expand(ENV, P.PlanParams.from_envelope(ENV), optional=True)


def test_full_grid_counts_per_arm(full_grid):
    c = P.counts(full_grid)["per_arm"]
    assert {a: v["total"] for a, v in c.items()} == {
        "A0": 14850, "A1": 27045, "A2p": 12600, "A2c": 36060, "A3": 1890, "A4": 1533, "A6": 630, "A3d": 1080}
    assert c["A0"]["by_tag"] == {"gate_matched": 5400, "main": 9450}
    assert c["A3d"]["tier"] == "optional"
    # what is runnable today (S6): A1 / A2c; the penalty arms wait for lambda*, A3 / A4 / A6 for S7 / S8 (+ S9)
    assert {a: v["runnable"] for a, v in c.items()} == {
        "A0": 0, "A1": 27045, "A2p": 0, "A2c": 36060, "A3": 0, "A4": 0, "A6": 0, "A3d": 0}
    assert c["A3"]["placeholder_reasons"] == {"arm_not_registered,lambda_star": 1890}
    assert c["A4"]["placeholder_reasons"] == {"arm_not_registered,recursion_cap": 1533}
    assert c["A0"]["placeholder_reasons"] == {"lambda_star": 9450, "lambda_star,gate_matched_depth": 5400}


def test_full_grid_cells(full_grid):
    pc = P.counts(full_grid)["per_cell"]
    confined = [r for r in pc if not r.startswith("penalty")]
    assert len(confined) == 21
    assert pc["ring|violation|K24|N5"]["A1"]["total"] == 10 * 3 * 3 * 5       # 10 k24 draws x 3 q x 3 L x R5
    assert pc["ring|violation|K24|N6"]["A2c"]["total"] == 21 * 3 * 20
    assert pc["ring|violation|K12|N7"]["A4"]["total"] == 90
    assert "A4" not in pc["complete|violation|K12|N4"]                         # no connectivity axis for A4
    assert sum("A4" in v for v in pc.values()) == 18
    assert pc["penalty|N4"]["A0"]["total"] == 90 * (3 + 3) * 5                 # 5 / 7 / 9 + three matched depths
    assert pc["penalty|N8"]["A0"]["total"] == 90 * 3 * 5                       # no A1 at N >= 8: no matched depth
    assert pc["penalty|N10"]["A6"]["total"] == 90 and "A3d" not in pc["penalty|N8"]
    a4 = [s for s in full_grid if s["arm"] == "A4"]
    assert {s["cell"]["connectivity"] for s in a4} == {"adaptive"}


def test_plan_runids_reproduce_stored_s4_runs():
    """The plan computes run_ids through the arms' own config: the S4 smoke runs (N04e004q1.5, lam 0.005) are found."""
    if not (GSP_ROOT / "results" / "runs" / "A2c").exists():
        pytest.skip("no S4 runs in this results root")
    flt = P.PlanFilter(N=(4,), draws=1, q=(1.5,), arms=("A0", "A1", "A2p", "A2c"))
    specs = P.build_plan(ENV, P.PlanParams.from_envelope(ENV, lam_star={4: 0.005}), flt)
    have = {a: sum((run_dir(a, s["run_id"]) / "run.json").exists() for s in specs if s["arm"] == a and s["run_id"])
            for a in ("A0", "A1", "A2p", "A2c")}
    tot = {a: sum(s["arm"] == a for s in specs) for a in have}
    assert tot == {"A0": 30, "A1": 75, "A2p": 20, "A2c": 100}             # A0: 15 + 15 gate-matched placeholders
    assert have == {"A0": 7, "A1": 19, "A2p": 20, "A2c": 100}


def test_s9_inputs_make_specs_runnable_and_merge_duplicates():
    flt = P.PlanFilter(arms=("A0",), N=(4,), draws=1, q=(1.0,), restarts=1)
    ph = P.build_plan(ENV, P.PlanParams.from_envelope(ENV), flt)
    assert len(ph) == 6 and all(s["placeholder"] for s in ph)
    params = P.PlanParams.from_envelope(ENV, lam_star={4: 0.05}, gate_matched={4: {5: 7, 7: 45, 9: 61}})
    specs = P.build_plan(ENV, params, flt)
    # L0(4, 5) = 7 equals the main depth 7: one configuration, both tags
    assert sorted(s["effort"] for s in specs) == [5, 7, 9, 45, 61]
    assert [s["tag"] for s in specs if s["effort"] == 7] == ["main+gate_matched"]
    assert all(s["run_id"] and s["kw"]["lam"] == 0.05 for s in specs)


def test_registry_driven_new_arm(monkeypatch):
    """An arm registered in arms.base (as S7 will register A3) becomes runnable with no change to the plan code;
    kwargs its config does not take are dropped, and a dropped kwarg that collapses specs is an error."""
    import gsp.arms.base as base

    class A3Stub(FakeArm):
        name = "A3"

        def config(self, inst, cell, effort, seed, *, lam=None, adhoc=False, root=None):   # no `restart`
            return super().config(inst, cell, effort, seed, lam=lam, adhoc=adhoc, root=root)

    orig = base._arm_table
    monkeypatch.setattr(base, "_arm_table", lambda: {**orig(), "A3": A3Stub})
    flt = P.PlanFilter(arms=("A3",), N=(4,), draws=1, q=(1.0,))
    specs = P.build_plan(ENV, P.PlanParams.from_envelope(ENV, lam_star={4: 0.005}), flt)
    assert len(specs) == 3 and all(s["run_id"] and not s["placeholder"] for s in specs)
    assert all(s["kw"] == {"lam": 0.005} for s in specs)
    env2 = dict(ENV, arms=dict(ENV["arms"], A3=dict(ENV["arms"]["A3"], restarts=2)))
    with pytest.raises(P.PlanError, match="hash to one run_id"):
        P.build_plan(env2, P.PlanParams.from_envelope(env2, lam_star={4: 0.005}), flt)


def test_pilot_set():
    specs = P.build_plan(ENV, None, None, pilot=True)
    assert len(specs) == 4 * 10 * 5 + 3 * 5 * 5
    assert {s["effort"] for s in specs} == {7} and {s["q"] for s in specs} == {1.5}
    assert {s["restart"] for s in specs} == {0} and all(s["run_id"] for s in specs)


def test_queue_file_roundtrip_and_order(tmp_path):
    flt = P.PlanFilter(arms=("A1", "A2c"), N=(4, 5), draws=1, q=(1.0,), axes=("baseline",), efforts=(5,),
                       restarts=2, schedules=("primary",))
    specs = P.build_plan(ENV, None, flt)
    assert [(s["N"], s["arm"]) for s in specs] == [(4, "A1"), (4, "A1"), (4, "A2c"), (5, "A1"), (5, "A1"), (5, "A2c")]
    side = P.write_queue(specs, tmp_path / "q.jsonl", {"x": 1})
    back = P.read_queue(tmp_path / "q.jsonl")
    assert side["n_runs"] == 6 and [s["run_id"] for s in back] == [s["run_id"] for s in specs]
    assert json.loads(P.sidecar_path(tmp_path / "q.jsonl").read_text())["x"] == 1


def test_cli_plan_refuses_unfiltered_queue_from_draft_envelope(tmp_path, capsys):
    from gsp.cli import main
    assert ENV["status"] != "approved"
    assert main(["plan", "--out", str(tmp_path / "all.jsonl")]) == 2
    assert not (tmp_path / "all.jsonl").exists()
    assert main(["plan", "--arms", "A2c", "--N", "4", "--draws", "1", "--q", "1.0", "--efforts", "5",
                 "--axes", "baseline", "--out", str(tmp_path / "one.jsonl")]) == 0
    assert len(P.read_queue(tmp_path / "one.jsonl")) == 2
    out = capsys.readouterr().out
    assert "full grid" in out and "A0 14850" in out and "A2c 36060" in out


# --- the runner (fake arm, in process) ----------------------------------------------------------------------------
@pytest.fixture
def fakes(monkeypatch):
    install_fakes(monkeypatch.setattr)


def _runner(q, out, **kw):
    kw.setdefault("arm_factory", factory())
    return Runner(q, None, out, gpu=False, aggregate=False, echo=False, heartbeat_s=0.05, **kw)


def test_runner_runs_skips_done_and_continues_past_failures(tmp_path, fakes):
    specs = [fake_spec(1), fake_spec(2), fake_spec(3), dict(fake_spec(4), arm="AX", run_id=None), fake_spec(1)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    r = _runner(q, out, arm_factory=factory(fail={2})).run()
    c = r["counts"]
    assert r["exit_code"] == 1 and r["state"] == "finished"
    assert (c["unique"], c["duplicates_in_queue"], c["done_now"], c["failed_now"]) == (4, 1, 2, 2)
    st, rec = run_state(FAKE, specs[1]["run_id"], out)
    assert st == "failed" and "planned failure at effort 2" in rec["error"]
    fails = [json.loads(x) for x in (out / "logs" / "queue_q.failures.jsonl").read_text().splitlines()]
    assert [f["arm"] for f in fails] == [FAKE, "AX"]
    assert "RuntimeError" in fails[0]["error"] and "KeyError" in fails[1]["error"]
    assert "--- FAILED" in run_log_path(FAKE, specs[1]["run_id"], out).read_text()
    for s in (specs[0], specs[2]):
        assert run_state(FAKE, s["run_id"], out)[0] == "done"
        assert postrun_state(run_dir(FAKE, s["run_id"], out)) == "finalized"
    started = run_state(FAKE, specs[0]["run_id"], out)[1]["started_at"]
    # retry_failed=False skips the failed run (AX has no run.json, so it is attempted again and fails again)
    rb = _runner(q, out, retry_failed=False).run()
    assert (rb["counts"]["skipped_failed"], rb["counts"]["attempted"], rb["counts"]["failed_now"]) == (1, 1, 1)
    # resume: done runs are skipped (not re-run), the failed one is retried and now succeeds
    r2 = _runner(q, out).run()
    assert r2["counts"]["done_before"] == 2 and r2["counts"]["done_now"] == 1 and r2["counts"]["failed_now"] == 1
    assert run_state(FAKE, specs[0]["run_id"], out)[1]["started_at"] == started
    assert run_state(FAKE, specs[1]["run_id"], out)[0] == "done"
    assert run_log_path(FAKE, specs[1]["run_id"], out).read_text().count("=== attempt") == 2
    assert run_log_path(FAKE, specs[0]["run_id"], out).read_text().count("=== attempt") == 1


def test_sigterm_mid_run_marks_failed_and_resumes(tmp_path, fakes):
    specs = [fake_spec(1), fake_spec(2), fake_spec(3)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    h0 = signal.getsignal(signal.SIGTERM)
    r = _runner(q, out, arm_factory=factory(steps=6, dt=0.02, self_signal=signal.SIGTERM, signal_efforts={2})).run()
    assert signal.getsignal(signal.SIGTERM) is h0                          # handlers restored
    assert r["exit_code"] == 128 + signal.SIGTERM and r["state"] == "stopped" and r["stop_signal"] == "SIGTERM"
    assert run_state(FAKE, specs[0]["run_id"], out)[0] == "done"
    st, rec = run_state(FAKE, specs[1]["run_id"], out)
    assert st == "failed" and "QueueStop" in rec["error"] and "SIGTERM" in rec["error"]
    assert run_state(FAKE, specs[2]["run_id"], out)[0] == "absent"          # never started
    pr = json.loads(progress_path("q", out).read_text())
    assert pr["state"] == "stopped" and pr["interrupted_run"]["run_id"] == specs[1]["run_id"]
    r2 = _runner(q, out).run()
    assert r2["exit_code"] == 0 and r2["counts"]["done_now"] == 2 and r2["counts"]["done_before"] == 1
    assert all(run_state(FAKE, s["run_id"], out)[0] == "done" for s in specs)
    assert len(list((out / "runs" / FAKE).iterdir())) == 3


def test_sigusr1_is_graceful(tmp_path, fakes):
    specs = [fake_spec(1), fake_spec(2), fake_spec(3)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    r = _runner(q, out, arm_factory=factory(steps=4, dt=0.01, self_signal=signal.SIGUSR1, signal_efforts={1})).run()
    assert r["exit_code"] == 0 and r["state"] == "stopped" and r["counts"]["done_now"] == 1
    assert run_state(FAKE, specs[0]["run_id"], out)[0] == "done"
    assert run_state(FAKE, specs[1]["run_id"], out)[0] == "absent"


def test_stale_running_other_host_and_postrun_backfill(tmp_path, fakes):
    from gsp.store.records import new_record, write_record
    from gsp.instances.instance import load_instance
    specs = [fake_spec(1), fake_spec(2), fake_spec(3)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    inst = load_instance(specs[0]["inst_id"])
    for s, host in ((specs[0], None), (specs[1], "elsewhere")):
        rec = new_record(FakeArm().config(inst, None, s["effort"], None, **s["kw"]).to_dict())
        if host:
            rec["host"] = host
        write_record(rec, out)
        assert run_state(FAKE, s["run_id"], out)[0] == "running"
    # spec 3 done without its post-run files (a runner stopped between "done" and the post-run step)
    FakeArm().run(inst, None, 3, restart=0, root=None, runs_root=out, finalize=False)
    assert postrun_state(run_dir(FAKE, specs[2]["run_id"], out)) == "unfinalized"
    r = _runner(q, out).run()
    c = r["counts"]
    assert (c["done_now"], c["skipped_running_elsewhere"], c["postrun_backfilled"]) == (1, 1, 1)
    assert run_state(FAKE, specs[0]["run_id"], out)[0] == "done"            # this host's stale record: re-run
    assert run_state(FAKE, specs[1]["run_id"], out)[0] == "running"         # another host's: left alone
    assert postrun_state(run_dir(FAKE, specs[2]["run_id"], out)) == "finalized"
    r2 = _runner(q, out, steal=True).run()
    assert r2["counts"]["done_now"] == 1 and run_state(FAKE, specs[1]["run_id"], out)[0] == "done"


def test_plan_mismatch_lock_and_max_runs(tmp_path, fakes):
    import fcntl
    bad = dict(fake_spec(1), run_id="0" * 16)
    q = write_specs(tmp_path / "q.jsonl", [bad, fake_spec(2), fake_spec(3)])
    out = tmp_path / "res"
    r = _runner(q, out, max_runs=2).run()
    assert r["counts"]["attempted"] == 2 and r["counts"]["failed_now"] == 1 and r["counts"]["done_now"] == 1
    fails = (out / "logs" / "queue_q.failures.jsonl").read_text()
    assert "PlanMismatch" in fails and "stale queue" in fails
    assert not (out / "runs" / FAKE / ("0" * 16)).exists()
    lp = lock_path(out)
    fd = os.open(lp, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(QueueBusy, match="another queue runner"):
            _runner(q, out).run()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_progress_file_during_run(tmp_path, fakes):
    seen = {}
    out = tmp_path / "res"

    def probe(cfg):
        seen["progress"] = json.loads(progress_path("q", out).read_text())

    specs = [fake_spec(1), fake_spec(2)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    _runner(q, out, arm_factory=factory(probe=probe)).run()
    pr = seen["progress"]
    assert pr["state"] == "running" and pr["pid"] == os.getpid()
    assert pr["current"]["run_id"] == specs[1]["run_id"] and pr["current"]["phase"] == "execute"
    assert pr["counts"]["done_now"] == 1 and pr["last"]["run_id"] == specs[0]["run_id"]


# --- the real kill test, CPU: a subprocess stopped by SIGTERM, SIGINT (in the post-run pause) and SIGKILL ---------
def _wait_progress(out, pred, timeout=30):
    p = progress_path("q", out)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            pr = json.loads(p.read_text())
            if pred(pr):
                return pr
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    raise TimeoutError("progress condition not reached")


def test_subprocess_kill_resume_no_duplicates(tmp_path):
    specs = [fake_spec(e, r) for e in (1, 2, 3) for r in (0, 1)]
    q = write_specs(tmp_path / "q.jsonl", specs)
    out = tmp_path / "res"
    drv = [sys.executable, str(GSP_ROOT / "tests" / "helpers" / "fake_runner.py"), str(q), str(out)]
    env = dict(os.environ, FAKE_STEPS="40", FAKE_DT="0.05")

    # 1. SIGTERM while the 2nd run executes
    p = subprocess.Popen(drv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cur = _wait_progress(out, lambda pr: (pr.get("current") or {}).get("index") == 1
                         and pr["current"].get("phase") == "execute" and pr["current"].get("elapsed_s", 0) > 0.2)
    p.send_signal(signal.SIGTERM)
    assert p.wait(30) == 128 + signal.SIGTERM
    rid = cur["current"]["run_id"]
    st, rec = run_state(FAKE, rid, out)
    assert st == "failed" and "QueueStop: queue stopped by SIGTERM" in rec["error"]

    # 2. SIGINT in the pause between "done" and the post-run step
    p = subprocess.Popen(drv, env=dict(env, FAKE_PAUSE_POSTRUN="30"), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    cur = _wait_progress(out, lambda pr: (pr.get("current") or {}).get("phase") == "postrun")
    p.send_signal(signal.SIGINT)
    assert p.wait(30) == 128 + signal.SIGINT
    rid2 = cur["current"]["run_id"]
    assert rid2 == rid and run_state(FAKE, rid2, out)[0] == "done"
    assert postrun_state(run_dir(FAKE, rid2, out)) == "unfinalized"
    df = M.coverage(P.read_queue(q), out)
    assert df.set_index("run_id").loc[rid2, "state"] == "unfinalized"

    # 3. SIGKILL mid-run: the record stays "running" (stale), the lock dies with the process
    p = subprocess.Popen(drv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cur = _wait_progress(out, lambda pr: (pr.get("current") or {}).get("phase") == "execute"
                         and pr["current"].get("elapsed_s", 0) > 0.2)
    p.kill()
    p.wait(30)
    rid3 = cur["current"]["run_id"]
    assert run_state(FAKE, rid3, out)[0] == "running"

    # 4. resume to the end
    p = subprocess.run(drv, env=dict(env, FAKE_STEPS="2"), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["counts"]["done_before"] == 2 and res["counts"]["done_now"] == 4
    # the unfinalized run was completed by the 3rd runner, before it was killed
    qlog = (out / "logs" / "queue_q.log").read_text()
    assert qlog.count(f"{rid2} was done without its post-run files (unfinalized): finalize_run -> done") == 1
    assert f"{rid3} has a stale 'running' record" in qlog
    # no duplicates: one directory per planned run_id, all done and finalized
    dirs = sorted(d.name for d in (out / "runs" / FAKE).iterdir())
    assert dirs == sorted(s["run_id"] for s in specs)
    assert all(run_state(FAKE, s["run_id"], out)[0] == "done" for s in specs)
    assert all(postrun_state(run_dir(FAKE, s["run_id"], out)) == "finalized" for s in specs)
    assert run_log_path(FAKE, rid, out).read_text().count("=== attempt") == 2
    assert run_log_path(FAKE, rid3, out).read_text().count("=== attempt") == 2
    assert (M.coverage(P.read_queue(q), out)["state"] == "done").all()


# --- missing ----------------------------------------------------------------------------------------------------
def test_missing_summary_and_resume_queue(tmp_path, fakes):
    specs = [fake_spec(1), fake_spec(2), fake_spec(3)]
    q = write_specs(tmp_path / "q.jsonl", specs[:1])
    out = tmp_path / "res"
    _runner(q, out).run()
    allq = write_specs(tmp_path / "all.jsonl", specs)
    df = M.coverage(P.read_queue(allq), out)
    assert list(df["state"]) == ["done", "absent", "absent"]
    text = M.summarize(df)
    assert "3 planned runs: done 1" in text and "absent 2" in text and "N04e004q1.0: L2 [absent], L3 [absent]" in text
    assert M.write_missing_queue(df, tmp_path / "miss.jsonl") == 2
    assert [s["run_id"] for s in P.read_queue(tmp_path / "miss.jsonl")] == [specs[1]["run_id"], specs[2]["run_id"]]
    js = M.as_json(df)
    assert js["missing"] == 2 and js["states"]["done"] == 1


def test_missing_on_the_full_plan_counts_placeholders(full_grid):
    df = M.coverage([s for s in full_grid if s["arm"] in ("A3", "A6")][:50])
    assert set(df["state"]) == {"placeholder"}


# --- merge (remote shards) ----------------------------------------------------------------------------------------
def test_merge_rules(tmp_path, fakes):
    specs = [fake_spec(1), fake_spec(2), fake_spec(3), fake_spec(4)]
    shard, local = tmp_path / "shard", tmp_path / "local"
    _runner(write_specs(tmp_path / "a.jsonl", specs), shard).run()
    # local: spec 1 failed (to be replaced), spec 2 done differently (kept), spec 3 absent (copied)
    _runner(write_specs(tmp_path / "b.jsonl", [specs[0]]), local, arm_factory=factory(fail={1})).run()
    _runner(write_specs(tmp_path / "c.jsonl", [specs[1]]), local).run()
    rj = run_dir(FAKE, specs[3]["run_id"], shard) / "run.json"
    rec = json.loads(rj.read_text())
    rec["status"] = "running"
    rj.write_text(json.dumps(rec))                                         # shard spec 4: not done -> skipped
    bogus = shard / "runs" / FAKE / "ffffffffffffffff"
    bogus.mkdir()
    (bogus / "run.json").write_text(json.dumps(json.loads((run_dir(FAKE, specs[2]["run_id"], shard) /
                                                          "run.json").read_text())))
    out = MG.merge_runs(shard / "runs", local)
    assert (out["copied"], out["replaced"], out["not_done_skipped"], out["invalid_skipped"]) == (1, 1, 1, 1)
    assert out["same"] + out["conflict_kept_local"] == 1
    assert run_state(FAKE, specs[0]["run_id"], local)[0] == "done"
    assert run_state(FAKE, specs[2]["run_id"], local)[0] == "done"
    assert run_state(FAKE, specs[3]["run_id"], local)[0] == "absent"
    again = MG.merge_runs(shard / "runs", local)
    assert again["copied"] == again["replaced"] == 0


# --- scripts ------------------------------------------------------------------------------------------------------
def _fake_bin(tmp_path, body):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    f = b / "nvidia-smi"
    f.write_text("#!/usr/bin/env bash\n" + body + "\n")
    f.chmod(0o755)
    return f"{b}:{os.environ['PATH']}"


def test_queue_sh_guards(tmp_path):
    q = write_specs(tmp_path / "q.jsonl", [dict(fake_spec(1), arm="AX", run_id=None)])
    res = tmp_path / "res"
    base = dict(os.environ, GSP_RESULTS=str(res))
    sh = ["bash", str(SCRIPTS / "queue.sh")]
    r = subprocess.run(sh + [str(q)], env=dict(base, PATH=_fake_bin(tmp_path, 'echo "4242, python"')),
                       capture_output=True, text=True)
    assert r.returncode == 3 and "GPU is busy (4242, python)" in r.stderr
    r = subprocess.run(sh + [str(q)], env=dict(base, PATH=_fake_bin(tmp_path, "exit 9")), capture_output=True,
                       text=True)
    assert r.returncode == 3 and "nvidia-smi failed" in r.stderr
    assert subprocess.run(sh + [str(tmp_path / "nope.jsonl")], env=base, capture_output=True).returncode == 2
    assert not (res / "runs").exists()
    # the lock: a second launcher is refused while the first holds it
    import fcntl
    (res / "logs").mkdir(parents=True, exist_ok=True)
    fd = os.open(res / "logs" / "queue.sh.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        r = subprocess.run(sh + [str(q)], env=dict(base, GSP_SKIP_GPU_GUARD="1"), capture_output=True, text=True)
        assert r.returncode == 3 and "another queue process" in r.stderr
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    # the full path on the CPU: a queue whose only spec names an unregistered arm -> recorded failure, exit 1
    r = subprocess.run(sh + [str(q), "--no-aggregate"], env=dict(base, GSP_SKIP_GPU_GUARD="1"),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, r.stderr
    console = (res / "logs" / "queue_q.console.log").read_text()
    assert "queue.sh: pid" in console and '"failed_now": 1' in console
    assert "KeyError" in (res / "logs" / "queue_q.failures.jsonl").read_text()


def test_remote_scripts_local_only(tmp_path, fakes):
    dest = tmp_path / "remote"
    q = write_specs(tmp_path / "q.jsonl", [fake_spec(1)])
    r = subprocess.run(["bash", str(SCRIPTS / "remote" / "push.sh"), str(dest), str(q)], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    g = dest / "GSP"
    assert (g / "gsp" / "cli.py").exists() and (g / "scripts" / "queue.sh").exists()
    assert (g / "results" / "instances" / "seed_table.csv").exists()
    assert any((g / "results" / "sectors").glob("sectors_*.npz"))
    assert (g / "results" / "queues" / "q.jsonl").exists() and not (g / "results" / "runs").exists()
    for s in ("push.sh", "pull.sh"):
        r = subprocess.run(["bash", str(SCRIPTS / "remote" / s), "someone@host.invalid:/x"], capture_output=True,
                           text=True, env={k: v for k, v in os.environ.items() if k != "GSP_REMOTE_APPROVED"})
        assert r.returncode == 4 and "Sensei's explicit approval" in r.stderr
    # a "remote" store with one done run; pull it into a separate local results root
    _runner(q, g / "results").run()
    local = tmp_path / "local"
    r = subprocess.run(["bash", str(SCRIPTS / "remote" / "pull.sh"), str(dest), "hostA"], capture_output=True,
                       text=True, env=dict(os.environ, GSP_RESULTS=str(local)), timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["copied"] == 1 and json.loads(r.stdout)["indexed"] == 1
    assert (local / "incoming" / "hostA" / "logs" / "queue_q.log").exists()
    rid = fake_spec(1)["run_id"]
    assert run_state(FAKE, rid, local)[0] == "done"
