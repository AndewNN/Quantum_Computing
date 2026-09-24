"""S6 on the GPU: a small real queue (A2p, A2c and A1 at N = 4, from `gsp plan`) through the real arms, the S5
post-run step and the incremental aggregate, written to a temporary results root; a resume runs nothing."""

import pytest

from gsp.runner import missing as M
from gsp.runner import plan as P
from gsp.runner.queue import postrun_state, run_queue, run_state
from gsp.store.paths import run_dir


@pytest.mark.gpu
def test_real_queue_runs_finalizes_and_resumes(tmp_path):
    env = P.load_envelope()
    flt = P.PlanFilter(arms=("A2p", "A2c", "A1"), N=(4,), draws=1, q=(3.0,), axes=("baseline", "penalty"),
                       efforts=(5,), restarts=1, schedules=("primary",))
    specs = P.build_plan(env, P.PlanParams.from_envelope(env, lam_star={4: 0.005}), flt)
    assert [s["arm"] for s in specs] == ["A1", "A2p", "A2c"]
    q = tmp_path / "q.jsonl"
    P.write_queue(specs, q)
    out = tmp_path / "res"
    # gpu=False: the conftest has already checked the GPU is free, and this process may hold its own context
    r = run_queue(q, None, out, gpu=False, echo=False, heartbeat_s=1.0)
    assert r["exit_code"] == 0 and r["counts"]["done_now"] == 3
    for s in specs:
        assert run_state(s["arm"], s["run_id"], out)[0] == "done"
        assert postrun_state(run_dir(s["arm"], s["run_id"], out)) == "finalized"
    from gsp.store.index import load_metrics
    m = load_metrics(out)
    assert len(m) == 3 and (m["anomalies"] == "").all()
    r2 = run_queue(q, None, out, gpu=False, echo=False, aggregate=False)
    assert r2["counts"]["done_before"] == 3 and r2["counts"]["attempted"] == 0
    assert (M.coverage(P.read_queue(q), out)["state"] == "done").all()
