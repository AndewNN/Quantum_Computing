#!/usr/bin/env bash
# S4 GPU queue: one GPU process at a time (PLAN §3.3), each step logged. Run from anywhere:
#   bash GSP/scripts/s4_queue.sh [steps...]      (default: repro anneal smoke time report)
# Each step is idempotent (done runs are loaded from results/runs).
set -u
PY=~/anaconda3/envs/gsp/bin/python
GSP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$GSP_DIR/results/logs"
mkdir -p "$LOG_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(repro anneal smoke time report)
cd "$GSP_DIR"
for s in "${steps[@]}"; do
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 10; done
  echo "[$(date -u +%FT%TZ)] start $s" | tee -a "$LOG_DIR/s4_queue.log"
  if [ "$s" = "legacy" ]; then
    $PY scripts/s4_legacy_checks.py > "$LOG_DIR/s4_$s.log" 2>&1
  else
    $PY -m gsp.cli arms "$s" > "$LOG_DIR/s4_$s.log" 2>&1
  fi
  echo "[$(date -u +%FT%TZ)] end $s rc=$?" | tee -a "$LOG_DIR/s4_queue.log"
done
