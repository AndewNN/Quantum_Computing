"""Subprocess driver for the kill / resume test (S6): runs a queue of FakeArm specs through the real Runner with
its real signal handling, CPU only.  python fake_runner.py QUEUE OUT_ROOT  (env: FAKE_STEPS, FAKE_DT,
FAKE_PAUSE_POSTRUN)."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # GSP/, so `tests.helpers` imports

from tests.helpers.fake_arm import factory, install_fakes  # noqa: E402

install_fakes()
from gsp.runner.queue import run_queue  # noqa: E402

res = run_queue(sys.argv[1], root=None, out_root=sys.argv[2], gpu=False, aggregate=False, heartbeat_s=0.1,
                arm_factory=factory(steps=int(os.environ.get("FAKE_STEPS", "40")),
                                    dt=float(os.environ.get("FAKE_DT", "0.05"))),
                debug_pause_postrun=float(os.environ.get("FAKE_PAUSE_POSTRUN", "0")), echo=False)
print(json.dumps(res))
sys.exit(res["exit_code"])
