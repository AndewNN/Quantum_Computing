"""Harness version (PLAN §3.2).

`HARNESS_VERSION` is hashed into every run_id. Bump it (semver) only when the numbers a run
produces would change; commits that do not change numerics leave it alone.
"""

HARNESS_VERSION = "0.1.0"
