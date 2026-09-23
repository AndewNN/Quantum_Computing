"""Static checks: the draft envelope (PLAN §1.2) and the portability rule (PLAN §3.5)."""

import re
from collections import Counter

import yaml

from gsp.store.paths import GSP_ROOT, configs_dir


def test_envelope_draft_matches_plan():
    env = yaml.safe_load((configs_dir() / "envelope.yaml").read_text())
    assert env["status"] == "draft"
    cells = env["confined_cells"]
    assert len(cells) == 21
    key = Counter((c["axis"], c["connectivity"], c["rule"], c["K"]) for c in cells)
    assert key[("baseline", "ring", "violation", 12)] == 4
    assert key[("K", "ring", "violation", 6)] == 4 and key[("K", "ring", "violation", 8)] == 4
    assert key[("K", "ring", "violation", 24)] == 2
    assert key[("rule", "ring", "objective", 12)] == 4
    assert key[("connectivity", "complete", "violation", 12)] == 3
    assert sorted(c["N"] for c in cells if c["K"] == 24) == [5, 6]
    assert sorted(c["N"] for c in cells if c["connectivity"] == "complete") == [4, 5, 6]
    assert sum(c["connectivity"] == "ring" for c in cells) == 18      # A4's cells
    assert env["instances"]["q"] == [1.0, 1.5, 3.0]
    assert env["instances"]["total_instances"] == 7 * 30 * 3
    assert env["instances"]["K_req"] == 12
    k24 = [c for c in cells if c["K"] == 24]
    assert all(c["draws"] == "k24_eligible" for c in k24)
    assert all("draws" not in c for c in cells if c["K"] != 24)


def test_simulator_calls_only_in_backend():
    """set_target / observe / sample / get_state are called only in gsp/sim/backend.py: any other
    module that imports cudaq may not call them (under any alias) or import them by name."""
    call = re.compile(r"\.\s*(set_target|observe|sample|get_state)\s*\(")
    named = re.compile(r"from\s+cudaq\s+import\s+.*\b(set_target|observe|sample|get_state)\b")
    offenders = []
    for p in (GSP_ROOT / "gsp").rglob("*.py"):
        if p.name == "backend.py" and p.parent.name == "sim":
            continue
        text = p.read_text()
        if "import cudaq" not in text and "from cudaq" not in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if call.search(line) or named.search(line):
                offenders.append(f"{p.relative_to(GSP_ROOT)}:{i}")
    assert not offenders, offenders


def test_kernels_only_in_circuits():
    offenders = [str(p.relative_to(GSP_ROOT)) for p in (GSP_ROOT / "gsp").rglob("*.py")
                 if "@cudaq.kernel" in p.read_text() and p.parent.name != "circuits"]
    assert not offenders, offenders
