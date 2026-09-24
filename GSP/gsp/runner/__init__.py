"""Plan, queue, coverage and merge of runs (S6): `gsp plan | run | missing | progress | merge | report`.

plan.py      the OFAT envelope (PLAN §1.2) -> run specs / queue files, counts per arm and cell
queue.py     the sequential, resumable queue runner (per-run logs, heartbeat, clean stop on SIGTERM / SIGINT)
missing.py   which cells / instances / efforts of a plan or queue lack done runs
merge.py     fold a pulled remote results/runs shard into the local store (never overwrites a done run)
report.py    the harness report stub (S15 builds it out)
"""
