"""A CPU-only arm for the queue-runner tests (S6): no cudaq, no GPU.

`FakeArm.execute` sleeps `steps` x `dt` seconds (long enough to be killed), optionally signals its own process
halfway (`self_signal` on the efforts in `signal_efforts`), raises on the efforts in `fail`, and otherwise returns a
valid PLAN §1.6 trajectory. `install_fakes(setattr)` swaps the backend's target / runtime calls and the post-run
step for CPU stand-ins (`setattr` = pytest's monkeypatch.setattr, or plain setattr in a subprocess driver).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from gsp.arms.base import Arm, Outcome, RunConfig, make_arm, make_extras

FAKE = "AF"
INST = "N04e004q1.0"


class FakeArm(Arm):
    name = FAKE
    encoding = "penalty"
    effort_kind = "depth"

    def __init__(self, steps: int = 0, dt: float = 0.05, fail=(), self_signal=None, signal_efforts=(), probe=None):
        self.steps, self.dt, self.fail = int(steps), float(dt), set(fail)
        self.self_signal, self.signal_efforts, self.probe = self_signal, set(signal_efforts), probe

    def config(self, inst, cell, effort, seed, *, restart: int = 0, lam=None, adhoc: bool = False, root=None):
        return RunConfig(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                         effort=int(effort), restart=int(restart), lam=None if lam is None else float(lam),
                         seed=1000 + int(restart), extras=make_extras(fake=True))

    def execute(self, cfg, inst, rulers, cell, logger=True, root=None):
        if self.probe is not None:
            self.probe(cfg)
        for k in range(self.steps):
            if self.self_signal is not None and cfg.effort in self.signal_efforts and k == self.steps // 2:
                os.kill(os.getpid(), self.self_signal)
            time.sleep(self.dt)
        if cfg.effort in self.fail:
            raise RuntimeError(f"planned failure at effort {cfg.effort}")
        T = 3
        z = np.zeros(T)
        traj = {"t": np.arange(T), "params": np.zeros((T, 2)), "energy": z, "ar_f": np.full(T, 0.5),
                "p_feas": np.ones(T), "eps_tilde": z, "p_opt": z, "p_top10": z, "circuits_charged": np.arange(T),
                "g2q_ii": np.arange(T), "g2q_iii": np.arange(T), "wall": z}
        return Outcome(metrics={"ar_f": 0.5, "p_feas": 1.0}, timings={"fake_s": 0.0}, diagnostics={"n": inst.n},
                       trajectory=traj, counts={"per_circuit": {"cx_ii": 0, "cx_iii": 0}}, final_state=None)


def fake_finalize_run(run_dir, root=None, shots=1000, final_state=None, force=False, engine=None, catch=False):
    from gsp.store.io import save_npz
    run_dir = Path(run_dir)
    if not force and (run_dir / "postrun.json").exists() and (run_dir / "samples.npz").exists():
        return json.loads((run_dir / "postrun.json").read_text())
    save_npz(run_dir / "samples.npz", {"idx": np.array([0]), "counts": np.array([shots])})
    out = {"postrun_version": 1, "status": "done", "fake": True}
    (run_dir / "postrun.json").write_text(json.dumps(out))
    return out


def install_fakes(set_attr=setattr):
    import gsp.metrics.postrun as postrun
    import gsp.sim.backend as backend
    set_attr(backend, "ensure_target", lambda: None)
    set_attr(backend, "runtime_info", lambda: {})
    set_attr(postrun, "finalize_run", fake_finalize_run)


def factory(**kw):
    def f(name):
        return FakeArm(**kw) if name == FAKE else make_arm(name)
    return f


def fake_spec(effort: int, restart: int = 0, root=None, inst_id: str = INST, run_id=True) -> dict:
    from gsp.instances.instance import load_instance
    kw = {"restart": restart}
    s = {"arm": FAKE, "inst_id": inst_id, "cell": None, "effort": effort, "kw": kw, "cell_label": "penalty|N4",
         "N": 4, "restart": restart, "effort_kind": "depth"}
    if run_id:
        s["run_id"] = FakeArm().config(load_instance(inst_id, root), None, effort, None, **kw).run_id
    return s


def write_specs(path, specs):
    Path(path).write_text("".join(json.dumps(s) + "\n" for s in specs))
    return path
