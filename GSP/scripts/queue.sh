#!/usr/bin/env bash
# GSP queue launcher (PLAN §3.3, §5 S6): one queue process at a time on the local GPU.
#
#   bash GSP/scripts/queue.sh QUEUE.jsonl [gsp run options ...]
#   nohup bash GSP/scripts/queue.sh GSP/results/queues/NAME.jsonl >/dev/null 2>&1 &      # detached
#
# Refuses to start (exit 3) when
#   * another queue.sh holds results/logs/queue.sh.lock (flock; the runner also takes results/logs/queue.lock), or
#   * nvidia-smi lists any compute process (another CUDA-Q job holds the GPU), or nvidia-smi itself fails.
#     GSP_SKIP_GPU_GUARD=1 skips this check (and passes --no-gpu-guard to the runner): machines without nvidia-smi,
#     and the CPU-only tests of this script.
# Exit 2 on a usage error. Otherwise it exec's `python -m gsp.cli run --queue QUEUE ...` (same pid), whose exit code
# it returns: 0 finished, 1 finished with failed runs, 130 / 143 stopped by SIGINT / SIGTERM.
#
# Logs (results root = $GSP_RESULTS or GSP/results):
#   logs/queue_NAME.console.log    stdout + stderr of the runner (appended; crashes and C++ messages land here)
#   logs/queue_NAME.log            the runner's event log;  logs/queue_NAME.progress.json  the heartbeat
#   logs/runs/{arm}/{run_id}.log   one block per attempt of a run;  logs/queue.sh.pid  the pid of the last launch
# Stop: `kill -TERM <pid>` or `kill -INT <pid>` (the in-flight run is marked failed, resumable); `kill -USR1 <pid>`
# stops after the current run. Resume: the same command. Watch: `gsp progress NAME`; gaps: `gsp missing --queue Q`.
set -u
PY=${GSP_PY:-$HOME/anaconda3/envs/gsp/bin/python}
GSP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS=${GSP_RESULTS:-$GSP_DIR/results}
LOG_DIR="$RESULTS/logs"

if [ $# -lt 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  [ $# -lt 1 ] && exit 2 || exit 0
fi
QUEUE=$1
shift
if [ ! -f "$QUEUE" ]; then
  echo "queue.sh: no such queue file: $QUEUE" >&2
  exit 2
fi
QUEUE=$(readlink -f "$QUEUE")
NAME=$(basename "$QUEUE" .jsonl)
mkdir -p "$LOG_DIR"

# one queue.sh at a time: the lock is held by fd 9, which the exec'd runner inherits (released when it exits)
exec 9>>"$LOG_DIR/queue.sh.lock"
if ! flock -n 9; then
  echo "queue.sh: refused: another queue process holds $LOG_DIR/queue.sh.lock (last pid $(cat "$LOG_DIR/queue.sh.pid" 2>/dev/null || echo ?))" >&2
  exit 3
fi

EXTRA=()
if [ "${GSP_SKIP_GPU_GUARD:-0}" = "1" ]; then
  EXTRA+=(--no-gpu-guard)
else
  if ! SMI=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1); then
    echo "queue.sh: refused: nvidia-smi failed, so the GPU cannot be checked ($SMI); GSP_SKIP_GPU_GUARD=1 overrides" >&2
    exit 3
  fi
  BUSY=$(printf '%s\n' "$SMI" | sed '/^[[:space:]]*$/d' | paste -sd ';' -)
  if [ -n "$BUSY" ]; then
    echo "queue.sh: refused: the GPU is busy ($BUSY); one GPU process at a time (PLAN §3.3)" >&2
    exit 3
  fi
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CONSOLE="$LOG_DIR/queue_${NAME}.console.log"
echo $$ > "$LOG_DIR/queue.sh.pid"
echo "[$(date -u +%FT%TZ)] queue.sh: pid $$ on $(hostname) runs $QUEUE $*" >> "$CONSOLE"
echo "queue.sh: pid $$ runs $NAME; console $CONSOLE; watch with: gsp progress $NAME" >&2
cd "$GSP_DIR" || exit 2
exec "$PY" -m gsp.cli run --queue "$QUEUE" ${EXTRA[@]+"${EXTRA[@]}"} "$@" >> "$CONSOLE" 2>&1
