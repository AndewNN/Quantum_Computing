"""`gsp run --queue FILE`: the sequential, resumable queue runner (PLAN §3.2, §3.3; S6).

One process runs the specs of a queue file (`gsp plan`) in order through `Arm.run(..., finalize=True)` (the S5
post-run step included), then rebuilds the registry and runs the incremental `gsp aggregate`.

* **Idempotent / resumable.** A spec whose run.json says "done" is skipped (run_id = sha256 of the config, so a
  resumed queue never runs a configuration twice). A done run that lacks its post-run files (stopped between
  "done" and the post-run step) is completed here with `finalize_run`. "failed" runs are retried (unless
  retry_failed=False). A "running" record on this host is stale (a killed runner: the queue lock guarantees no
  other runner is active on this store) and is re-run; one written by another host is skipped unless steal=True.
* **Failures** (a spec that raises) are recorded -- run.json status "failed" with the traceback when the arm got as
  far as writing its record, and always in `logs/queue_{name}.failures.jsonl` and the per-run log -- and the
  queue continues.
* **Stop.** SIGTERM / SIGINT stop at once: the in-flight run is interrupted, `Arm.run` marks it "failed" (traceback
  ends in QueueStop), the progress file says "stopped", the exit code is 128 + signum; resuming re-runs it.
  Between runs a signal just ends the loop. SIGUSR1 = graceful: finish the current run, then stop. SIGKILL leaves
  the in-flight run "running", which the next runner treats as stale.
* **Monitoring.** `logs/queue_{name}.progress.json` (atomic rewrite at every event and every `heartbeat_s`
  seconds from a heartbeat thread): pid, host, state, counts, the current run and its phase (setup / execute /
  postrun), elapsed, rate and ETA. `logs/queue_{name}.log` has one line per event; `logs/runs/{arm}/{run_id}.log`
  one block per attempt (spec, Python-level stdout / stderr of the run, status, metrics or traceback).
* **Dynamic mode** (`dynamic=True`, `gsp run --dynamic`; TQE rerun): the queue file is re-read before every run
  and the first line (file order = priority) that is not done, not tried in this session and not running on
  another host is run next. Lines may be added, removed or reordered at any time (`gsp q ...`, atomic rewrites)
  without stopping the runner; the in-flight run is never touched. `follow=True` waits for new lines when the
  queue is exhausted (poll every `poll_s` seconds) instead of finishing. A spec may carry `seed` (the restart seed
  written at plan time, for ad hoc instances outside the seed table).
* **One process at a time.** An exclusive `flock` on `logs/queue.lock` of the output store, and (gpu=True) the
  nvidia-smi guard: refuse to start while another process holds a CUDA context.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from ..store.io import atomic_write_bytes
from ..store.paths import logs_dir, results_root, run_dir, run_log_path
from .plan import read_queue

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_BUSY = 0, 1, 2, 3
PROGRESS_SCHEMA = 1
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
GRACEFUL_SIGNAL = signal.SIGUSR1


class QueueStop(BaseException):
    """Raised inside the in-flight run by SIGTERM / SIGINT (a BaseException, so `Arm.run` marks the run failed and
    re-raises, and no `except Exception` on the way swallows it)."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__(f"queue stopped by {signal.Signals(self.signum).name}")


class QueueBusy(RuntimeError):
    pass


class PlanMismatch(RuntimeError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def queue_name(path) -> str:
    return Path(path).stem


def progress_path(name: str, out_root=None) -> Path:
    return logs_dir(out_root) / f"queue_{name}.progress.json"


def queue_log_path(name: str, out_root=None) -> Path:
    return logs_dir(out_root) / f"queue_{name}.log"


def failures_path(name: str, out_root=None) -> Path:
    return logs_dir(out_root) / f"queue_{name}.failures.jsonl"


def lock_path(out_root=None) -> Path:
    return logs_dir(out_root) / "queue.lock"


# --- store state -----------------------------------------------------------------------------------------------
def postrun_state(path) -> str:
    """"finalized" | "unfinalized" (post-run files missing) | "postrun_failed"."""
    from ..metrics.postrun import POSTRUN_FILE, SAMPLES_FILE
    path = Path(path)
    p = path / POSTRUN_FILE
    if not p.exists():
        return "unfinalized"
    try:
        st = json.loads(p.read_text()).get("status")
    except (OSError, json.JSONDecodeError):
        return "postrun_failed"
    if st != "done":
        return "postrun_failed"
    return "finalized" if (path / SAMPLES_FILE).exists() else "unfinalized"


def run_state(arm: str, rid: str, out_root=None) -> tuple[str, dict | None]:
    """(status, run.json) of a run in the store: absent | running | done | failed | unreadable."""
    p = run_dir(arm, rid, out_root) / "run.json"
    if not p.exists():
        return "absent", None
    try:
        rec = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return "unreadable", None
    return str(rec.get("status")), rec


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False
    return True


# --- the runner ------------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, queue_path, root=None, out_root=None, *, retry_failed: bool = True, steal: bool = False,
                 gpu: bool = True, finalize: bool = True, aggregate: bool = True, heartbeat_s: float = 30.0,
                 max_runs: int | None = None, arm_factory=None, debug_pause_postrun: float = 0.0,
                 echo=True, dynamic: bool = False, follow: bool = False, poll_s: float = 30.0):
        self.queue_path = Path(queue_path)
        self.name = queue_name(queue_path)
        self.root = root                                   # instances / sectors / seed table
        self.out_root = results_root(out_root if out_root is not None else root)
        self.retry_failed, self.steal, self.gpu, self.finalize = retry_failed, steal, gpu, finalize
        self.do_aggregate, self.heartbeat_s, self.max_runs = aggregate, float(heartbeat_s), max_runs
        self.debug_pause_postrun = float(debug_pause_postrun)
        self.echo = echo
        self.dynamic, self.follow, self.poll_s = bool(dynamic or follow), bool(follow), float(poll_s)
        self._done_ids: set = set()
        self._tried: set = set()
        if arm_factory is None:
            from ..arms.base import make_arm
            arm_factory = make_arm
        self.arm_factory = arm_factory
        self.host = socket.gethostname()
        self.stop_signal: int | None = None
        self.graceful = False
        self.armed = False
        self._lock_fd = None
        self._plock = threading.Lock()
        self._hb_stop = threading.Event()
        self._hb = None
        self._insts: dict = {}
        self._old_handlers: dict = {}
        self.progress: dict = {}

    # -- logging / progress
    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        p = queue_log_path(self.name, self.out_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, file=sys.__stdout__, flush=True)

    def write_progress(self) -> None:
        with self._plock:
            pr = self.progress
            pr["updated_at"] = _now()
            cur = pr.get("current")
            if cur and cur.get("_t0") is not None:
                cur["elapsed_s"] = round(time.time() - cur["_t0"], 3)
            done_now = pr["counts"]["done_now"]
            walls = pr.get("_walls", [])
            if walls:
                pr["mean_run_s"] = round(sum(walls) / len(walls), 3)
                pr["eta_s"] = round(pr["mean_run_s"] * max(pr["counts"]["remaining"], 0), 1)
            pr["counts"]["done_total"] = pr["counts"]["done_before"] + done_now
            out = {k: v for k, v in pr.items() if not k.startswith("_")}
            if out.get("current"):
                out["current"] = {k: v for k, v in out["current"].items() if not k.startswith("_")}
            atomic_write_bytes(progress_path(self.name, self.out_root),
                               json.dumps(out, indent=1, sort_keys=True, default=str).encode())

    def _heartbeat(self) -> None:
        while not self._hb_stop.wait(self.heartbeat_s):
            try:
                self.write_progress()
            except Exception:                       # pragma: no cover (never kill the queue for a heartbeat)
                pass

    # -- lock and guard
    def _acquire_lock(self) -> None:
        p = lock_path(self.out_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                holder = os.read(fd, 4096).decode(errors="replace").strip()
            finally:
                os.close(fd)
            raise QueueBusy(f"another queue runner holds {p}: {holder or '?'}") from None
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "host": self.host, "queue": str(self.queue_path),
                                 "started_at": _now()}).encode())
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                os.ftruncate(self._lock_fd, 0)
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def _gpu_guard(self) -> None:
        from ..sim import backend
        busy = backend.gpu_compute_pids()
        if busy:
            raise QueueBusy(f"GPU busy (compute PIDs {busy}); one GPU process at a time (PLAN §3.3)")
        backend.ensure_target()

    # -- signals
    def _handler(self, signum, frame):
        if signum == GRACEFUL_SIGNAL:
            self.graceful = True
            return
        if self.stop_signal is not None:
            return                                   # already stopping: the stop path is short, let it finish
        self.stop_signal = int(signum)
        self.progress["state"] = "stopping"
        self.progress["stop_signal"] = signal.Signals(signum).name
        if self.armed:
            self.armed = False
            raise QueueStop(signum)

    def _install_handlers(self) -> None:
        for s in STOP_SIGNALS + (GRACEFUL_SIGNAL,):
            self._old_handlers[s] = signal.signal(s, self._handler)

    def _restore_handlers(self) -> None:
        for s, h in self._old_handlers.items():
            signal.signal(s, h if h is not None else signal.SIG_DFL)
        self._old_handlers = {}

    @contextlib.contextmanager
    def _armed(self):
        if self.stop_signal is not None:
            raise QueueStop(self.stop_signal)
        self.armed = True
        try:
            yield
        finally:
            self.armed = False

    # -- helpers
    def _instance(self, inst_id):
        if inst_id not in self._insts:
            from ..instances.adhoc import load_any
            if len(self._insts) > 32:
                self._insts.clear()
            self._insts[inst_id] = load_any(inst_id, self.root)
        return self._insts[inst_id]

    def _set_current(self, i, spec, phase, reason=None):
        with self._plock:
            if phase is None:
                self.progress["current"] = None
                return
            cur = self.progress.get("current")
            if not cur or cur.get("index") != i:
                cur = {"index": i, "line": spec.get("_line"), "run_id": spec.get("run_id"), "arm": spec["arm"],
                       "inst_id": spec["inst_id"], "cell": spec.get("cell_label") or spec.get("cell"),
                       "effort": spec["effort"], "kw": spec.get("kw"), "started_at": _now(), "_t0": time.time(),
                       "attempt_reason": reason}
            cur["phase"] = phase
            self.progress["current"] = cur
        self.write_progress()

    def _on_done(self, i, spec):
        def cb(path, rec):
            self._set_current(i, spec, "postrun")
            if self.debug_pause_postrun > 0:
                time.sleep(self.debug_pause_postrun)
        return cb

    def _record_failure(self, i, spec, exc, run_status):
        tb = "".join(traceback.format_exception(exc))
        row = {"time": _now(), "index": i, "line": spec.get("_line"), "run_id": spec.get("run_id"),
               "arm": spec["arm"], "inst_id": spec["inst_id"], "cell": spec.get("cell"), "effort": spec["effort"],
               "kw": spec.get("kw"), "run_json_status": run_status, "error": tb}
        p = failures_path(self.name, self.out_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return tb

    # -- one spec
    def _process(self, i, spec) -> None:
        c = self.progress["counts"]
        rid, arm = spec.get("run_id"), spec["arm"]
        state, rec = run_state(arm, rid, self.out_root) if rid else ("absent", None)
        if state == "done":
            pst = postrun_state(run_dir(arm, rid, self.out_root))
            if not self.finalize or pst == "finalized":
                return
            self._set_current(i, spec, "postrun-backfill", reason=pst)
            from ..metrics import postrun
            with self._armed():
                out = postrun.finalize_run(run_dir(arm, rid, self.out_root), root=self.root, catch=True,
                                           force=(pst == "postrun_failed"))
            c["postrun_backfilled"] += 1
            self.log(f"#{i} {arm} {rid} was done without its post-run files ({pst}): finalize_run -> "
                     f"{out.get('status')}")
            self._set_current(i, spec, None)
            return
        if state == "running":
            host = (rec or {}).get("host")
            if host != self.host and not self.steal:
                c["skipped_running_elsewhere"] += 1
                self.log(f"#{i} {arm} {rid} is 'running' on {host}: skipped (steal=False)")
                return
            self.log(f"#{i} {arm} {rid} has a stale 'running' record (host {host}, started "
                     f"{(rec or {}).get('started_at')}): re-run")
        if state == "failed" and not self.retry_failed:
            c["skipped_failed"] += 1
            return
        if self.max_runs is not None and c["attempted"] >= self.max_runs:
            self.graceful = True
            return
        self._execute(i, spec, state)

    def _execute(self, i, spec, reason) -> None:
        c = self.progress["counts"]
        c["attempted"] += 1
        arm_name = spec["arm"]
        rid = spec.get("run_id") or "unresolved"
        logp = run_log_path(arm_name, rid, self.out_root)
        logp.parent.mkdir(parents=True, exist_ok=True)
        self._set_current(i, spec, "setup", reason=reason)
        t0 = time.time()
        self.log(f"#{i} start {arm_name} {rid} {spec['inst_id']} {spec.get('cell_label') or ''} "
                 f"effort={spec['effort']} kw={spec.get('kw')} ({reason})")
        status, err = None, None
        with open(logp, "a") as lf:
            lf.write(f"=== attempt {_now()} host={self.host} pid={os.getpid()} queue={self.queue_path} "
                     f"line={spec.get('_line')} reason={reason}\n"
                     f"spec: {json.dumps({k: v for k, v in spec.items() if not k.startswith('_')}, default=str)}\n")
            lf.flush()
            try:
                with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf), self._armed():
                    arm = self.arm_factory(arm_name)
                    inst, rul, adhoc = self._instance(spec["inst_id"])
                    kw = dict(spec.get("kw") or {})
                    seed = spec.get("seed")
                    cfg = arm.config(inst, spec.get("cell"), spec["effort"], seed, adhoc=adhoc, root=self.root, **kw)
                    if spec.get("run_id") and cfg.run_id != spec["run_id"]:
                        raise PlanMismatch(f"spec run_id {spec['run_id']} but the arm hashes it to {cfg.run_id} "
                                           "(stale queue: re-run `gsp plan`)")
                    rid = cfg.run_id
                    self._set_current(i, spec, "execute")
                    rr = arm.run(inst, spec.get("cell"), spec["effort"], seed, rulers=rul, root=self.root,
                                 runs_root=self.out_root, finalize=self.finalize, on_done=self._on_done(i, spec), **kw)
                status = rr.status
            except QueueStop as exc:
                st, _ = run_state(arm_name, rid, self.out_root)
                lf.write(f"--- STOPPED by {signal.Signals(exc.signum).name} after {time.time() - t0:.2f} s; "
                         f"run.json status: {st}\n{''.join(traceback.format_exception(exc))}\n")
                self.log(f"#{i} STOPPED {arm_name} {rid} by {signal.Signals(exc.signum).name} "
                         f"(run.json: {st}; resumable)")
                c["interrupted"] += 1
                self.progress["interrupted_run"] = {"index": i, "run_id": rid, "arm": arm_name, "run_json": st}
                raise
            except Exception as exc:                      # record, continue with the next spec
                st, _ = run_state(arm_name, rid, self.out_root) if rid != "unresolved" else ("absent", None)
                err = self._record_failure(i, spec, exc, st)
                lf.write(f"--- FAILED after {time.time() - t0:.2f} s; run.json status: {st}\n{err}\n")
                self.log(f"#{i} FAILED {arm_name} {rid}: {type(exc).__name__}: {exc} (run.json: {st})")
                c["failed_now"] += 1
                c["remaining"] = max(c["remaining"] - 1, 0)
                status = "failed"
            else:
                rec = rr.record
                wall = time.time() - t0
                pst = postrun_state(rr.path) if rr.path else None
                lf.write(f"--- {status} in {wall:.2f} s (run wall_s {rec.get('wall_s')}); postrun: {pst}; "
                         f"AR_F {rec.get('metric_ar_f')} p_feas {rec.get('metric_p_feas')}\n")
                self.log(f"#{i} {status} {arm_name} {rid} {wall:.2f} s AR_F={rec.get('metric_ar_f')} "
                         f"postrun={pst}{' (loaded)' if rr.skipped else ''}")
                c["remaining"] = max(c["remaining"] - 1, 0)
                if status == "done":
                    c["done_now"] += 1
                    self.progress["_walls"].append(wall)
                    self.progress["last"] = {"index": i, "run_id": rid, "arm": arm_name, "status": status,
                                             "wall_s": round(wall, 3), "ar_f": rec.get("metric_ar_f"),
                                             "postrun": pst, "finished_at": _now()}
            finally:
                self._set_current(i, spec, None)

    # -- the queue
    def _prescan(self, specs):
        c = {"total_lines": len(specs), "unique": 0, "duplicates_in_queue": 0, "done_before": 0,
             "unfinalized_before": 0, "failed_before": 0, "running_before": 0, "absent_before": 0,
             "done_now": 0, "failed_now": 0, "interrupted": 0, "attempted": 0, "postrun_backfilled": 0,
             "skipped_running_elsewhere": 0, "skipped_failed": 0, "remaining": 0}
        seen, out = set(), []
        for s in specs:
            key = s.get("run_id") or json.dumps([s["arm"], s["inst_id"], s.get("cell"), s["effort"], s.get("kw")],
                                                sort_keys=True)
            if key in seen:
                c["duplicates_in_queue"] += 1
                continue
            seen.add(key)
            out.append(s)
            st, _ = run_state(s["arm"], s["run_id"], self.out_root) if s.get("run_id") else ("absent", None)
            if st == "done":
                c["done_before"] += 1
                if postrun_state(run_dir(s["arm"], s["run_id"], self.out_root)) != "finalized":
                    c["unfinalized_before"] += 1
            elif st in ("failed", "running"):
                c[f"{st}_before"] += 1
            else:
                c["absent_before"] += 1
        c["unique"] = len(out)
        c["remaining"] = c["unique"] - c["done_before"]
        return out, c

    def _next_dynamic(self):
        """Re-read the queue file; return (line index, spec) of the first runnable line, or None. Updates the
        progress counts (unique / done_total / remaining) from the file as it is now."""
        try:
            specs = read_queue(self.queue_path)
        except Exception as exc:                       # a bad edit: keep the runner alive, retry at the next poll
            self.log(f"queue re-read FAILED ({type(exc).__name__}: {exc}); retry in {self.poll_s:g} s")
            return None, False
        c = self.progress["counts"]
        seen, pending, nxt = set(), 0, None
        for i, sp in enumerate(specs):
            rid = sp.get("run_id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            if rid in self._done_ids:
                continue
            st, rec = run_state(sp["arm"], rid, self.out_root)
            if st == "done":
                if not self.finalize or postrun_state(run_dir(sp["arm"], rid, self.out_root)) == "finalized":
                    self._done_ids.add(rid)
                    continue
            pending += 1
            if nxt is not None or rid in self._tried:
                continue
            if st == "running" and (rec or {}).get("host") != self.host and not self.steal:
                continue
            if st == "failed" and not self.retry_failed:
                continue
            nxt = (i, sp)
        c["unique"] = len(seen)
        c["remaining"] = pending
        c["done_before"] = len(seen) - pending - c["done_now"]
        return nxt, True

    def _run_dynamic(self):
        """The dynamic loop (see the module docstring). Returns the stop signal or None."""
        idle_logged = False
        while True:
            if self.stop_signal is not None:
                return self.stop_signal
            if self.graceful:
                return None
            nxt, ok = self._next_dynamic()
            if nxt is None:
                if ok and not self.follow:
                    return None
                if not idle_logged:
                    self.log(f"queue exhausted: waiting for new lines (poll {self.poll_s:g} s)")
                    idle_logged = True
                self.progress["state"] = "waiting"
                self.write_progress()
                t_end = time.time() + self.poll_s
                while time.time() < t_end and self.stop_signal is None and not self.graceful:
                    time.sleep(min(1.0, self.poll_s))
                continue
            if idle_logged:
                self.progress["state"] = "running"
                idle_logged = False
            i, spec = nxt
            self._tried.add(spec["run_id"])
            self.progress["position"] = i
            self._process(i, spec)
            if self.max_runs is not None and self.progress["counts"]["attempted"] >= self.max_runs:
                self.graceful = True

    def run(self) -> dict:
        t_start = time.time()
        specs = read_queue(self.queue_path)
        self._acquire_lock()
        try:
            if self.gpu:
                self._gpu_guard()
            specs, counts = self._prescan(specs)
            self.progress = {"schema": PROGRESS_SCHEMA, "queue": str(self.queue_path), "name": self.name,
                             "pid": os.getpid(), "host": self.host, "state": "running", "started_at": _now(),
                             "heartbeat_s": self.heartbeat_s, "out_root": str(self.out_root),
                             "counts": counts, "current": None, "last": None, "stop_signal": None,
                             "exit_code": None, "position": 0, "interrupted_run": None, "finished_at": None,
                             "mean_run_s": None, "eta_s": None, "_walls": []}
            self._install_handlers()
            self.write_progress()
            self.log(f"queue {self.queue_path} ({self.name}): pid {os.getpid()} on {self.host}; "
                     f"{counts['unique']} unique specs ({counts['duplicates_in_queue']} duplicate lines), "
                     f"{counts['done_before']} done before ({counts['unfinalized_before']} without post-run), "
                     f"{counts['failed_before']} failed, {counts['running_before']} running, "
                     f"{counts['absent_before']} absent")
            self._hb = threading.Thread(target=self._heartbeat, name="gsp-queue-heartbeat", daemon=True)
            self._hb.start()
            stopped = None
            try:
                for i, spec in enumerate([] if self.dynamic else specs):
                    if self.stop_signal is not None:
                        stopped = self.stop_signal
                        break
                    if self.graceful:
                        break
                    self.progress["position"] = i
                    self._process(i, spec)
                if self.dynamic:
                    stopped = self._run_dynamic()
                if self.stop_signal is not None:
                    stopped = self.stop_signal
                if stopped is None and not self.graceful and self.do_aggregate:
                    self._set_current(-1, {"arm": "-", "inst_id": "-", "effort": None}, "aggregate")
                    with self._armed():
                        self._aggregate()
                    self._set_current(-1, None, None)
            except QueueStop as exc:
                stopped = exc.signum
            finally:
                self._hb_stop.set()
                self._restore_handlers()
            c = self.progress["counts"]
            if stopped is not None:
                self.progress["state"] = "stopped"
                code = 128 + int(stopped)
                self.log(f"stopped by {signal.Signals(stopped).name}: {c['done_now']} done, {c['failed_now']} "
                         f"failed this session; resume with the same command")
            elif self.graceful:
                self.progress["state"] = "stopped"
                code = EXIT_OK if c["failed_now"] == 0 else EXIT_FAILED
                self.log(f"graceful stop: {c['done_now']} done, {c['failed_now']} failed this session")
            else:
                self.progress["state"] = "finished"
                code = EXIT_OK if c["failed_now"] == 0 else EXIT_FAILED
                self.log(f"finished in {time.time() - t_start:.1f} s: {c['done_now']} done now, "
                         f"{c['done_before']} before, {c['failed_now']} failed, {c['postrun_backfilled']} post-run "
                         f"backfilled, {c['skipped_running_elsewhere']} running elsewhere")
            self.progress["exit_code"] = code
            self.progress["current"] = None
            self.progress["finished_at"] = _now()
            self.write_progress()
            return {"exit_code": code, "state": self.progress["state"], "counts": dict(c),
                    "stop_signal": None if stopped is None else signal.Signals(stopped).name}
        finally:
            self._release_lock()

    def _aggregate(self) -> None:
        from ..store.index import build_index
        try:
            reg = build_index(self.out_root)
            from ..metrics.aggregate import aggregate
            df = aggregate(self.out_root, inputs_root=self.root)
            bad = int((df["anomalies"] != "").sum()) if not df.empty and "anomalies" in df else 0
            self.log(f"index: {len(reg)} runs; aggregate: {len(df)} rows, {bad} with anomalies")
        except QueueStop:
            raise
        except Exception as exc:
            self.log(f"aggregate FAILED (the runs are safe; run `gsp aggregate`): {type(exc).__name__}: {exc}")


def run_queue(queue_path, root=None, out_root=None, **kw) -> dict:
    return Runner(queue_path, root, out_root, **kw).run()


# --- monitoring ------------------------------------------------------------------------------------------------
def read_progress(name_or_path, out_root=None) -> dict:
    p = Path(name_or_path)
    if not p.suffix == ".json":
        p = progress_path(queue_name(name_or_path), out_root)
    pr = json.loads(p.read_text())
    try:
        upd = _dt.datetime.fromisoformat(pr["updated_at"])
        pr["heartbeat_age_s"] = round((_dt.datetime.now(_dt.timezone.utc) - upd).total_seconds(), 1)
    except Exception:
        pr["heartbeat_age_s"] = None
    local = pr.get("host") == socket.gethostname()
    pr["pid_alive"] = pid_alive(pr.get("pid")) if local else None
    if pr.get("state") in ("running", "stopping") and local and not pr["pid_alive"]:
        pr["state_note"] = "the runner is gone (killed?); its in-flight run is stale 'running' and resumes"
    return pr


def format_progress(pr: dict) -> str:
    c = pr.get("counts", {})
    cur = pr.get("current") or {}
    lines = [f"{pr.get('name')}: {pr.get('state')} (pid {pr.get('pid')} on {pr.get('host')}, alive "
             f"{pr.get('pid_alive')}, heartbeat {pr.get('heartbeat_age_s')} s ago)"
             + (f" -- {pr['state_note']}" if pr.get("state_note") else ""),
             f"  unique {c.get('unique')}: done {c.get('done_total')} ({c.get('done_before')} before, "
             f"{c.get('done_now')} now), failed now {c.get('failed_now')}, remaining {c.get('remaining')}, "
             f"post-run backfilled {c.get('postrun_backfilled')}"
             + (f", mean {pr.get('mean_run_s')} s/run, ETA {pr.get('eta_s')} s" if pr.get("mean_run_s") else "")]
    if cur:
        lines.append(f"  current #{cur.get('index')}: {cur.get('arm')} {cur.get('run_id')} {cur.get('inst_id')} "
                     f"effort={cur.get('effort')} phase={cur.get('phase')} elapsed {cur.get('elapsed_s')} s")
    if pr.get("last"):
        la = pr["last"]
        lines.append(f"  last: #{la.get('index')} {la.get('arm')} {la.get('run_id')} {la.get('status')} "
                     f"{la.get('wall_s')} s AR_F={la.get('ar_f')}")
    if pr.get("stop_signal"):
        lines.append(f"  stop signal: {pr['stop_signal']}; exit code {pr.get('exit_code')}")
    return "\n".join(lines)
