#!/usr/bin/env bash
# S9a launcher (PLAN §5 S9): the WP2 calibration queues, in priority order, one GPU process at a time.
#
#   nohup bash GSP/scripts/s9_calibration.sh >/dev/null 2>&1 &          # the production launch (detached)
#
# Runs, sequentially, through scripts/queue.sh (flock + nvidia-smi guard; each queue ends with index + aggregate):
#   1. results/queues/s9_q1_timing.jsonl     the timing set
#   2. results/queues/s9_q2_pilot.jsonl      the lambda pilot (`gsp plan --pilot`)
#   3. results/queues/s9_q3_evidence.jsonl   evidence for O-2 / O-11 (tagged runs, no rule reads them)
# then `gsp index` and `gsp aggregate` over the whole store.
#
# Resumable: re-launch with the same command; done runs are skipped by run_id, failed / interrupted runs are retried.
# Prints nothing: every line goes to results/logs/s9_calibration.log (queue.sh keeps its per-queue logs beside it:
# queue_NAME.console.log / .log / .progress.json / .failures.jsonl; watch with `gsp progress`).
#
# Exit codes: 0 every queue finished without failed runs; 1 finished, some runs failed (see the failures files;
# a re-launch retries them); 2 usage (a queue file is missing); 3 a guard refused (another launcher or queue holds the
# lock, the GPU stayed busy, nvidia-smi failed): nothing further is started; 130 / 143 stopped by SIGINT / SIGTERM
# (forwarded to the running queue, whose in-flight run is marked failed and resumes on re-launch); other: the exit
# code of the queue that crashed (the later queues are not started).
# Stop: kill -TERM "$(cat GSP/results/logs/s9_calibration.pid)"
#
# Environment (tests / smoke only): GSP_RESULTS = results root (default GSP/results); S9_QUEUES = space-separated
# queue files instead of the three above; S9_GPU_WAIT_S = how long to wait for the GPU to be free before each queue
# (default 300); GSP_PY = the python (default the gsp env).
set -u
PY=${GSP_PY:-$HOME/anaconda3/envs/gsp/bin/python}
GSP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS=${GSP_RESULTS:-$GSP_DIR/results}
LOG_DIR="$RESULTS/logs"
LOG="$LOG_DIR/s9_calibration.log"
GPU_WAIT_S=${S9_GPU_WAIT_S:-300}
mkdir -p "$LOG_DIR"
exec </dev/null >>"$LOG" 2>&1

log() { echo "[$(date -u +%FT%TZ)] s9_calibration: $*"; }

exec 8>>"$LOG_DIR/s9_calibration.lock"
if ! flock -n 8; then
  log "refused: another s9_calibration.sh holds $LOG_DIR/s9_calibration.lock (pid $(cat "$LOG_DIR/s9_calibration.pid" 2>/dev/null || echo ?))"
  exit 3
fi
echo $$ > "$LOG_DIR/s9_calibration.pid"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GSP_RESULTS="$RESULTS"
cd "$GSP_DIR" || exit 2

if [ -n "${S9_QUEUES:-}" ]; then
  read -r -a QUEUES <<< "$S9_QUEUES"
else
  QUEUES=("$RESULTS/queues/s9_q1_timing.jsonl" "$RESULTS/queues/s9_q2_pilot.jsonl" "$RESULTS/queues/s9_q3_evidence.jsonl")
fi
for q in "${QUEUES[@]}"; do
  if [ ! -f "$q" ]; then
    log "usage: no such queue file: $q"
    exit 2
  fi
done
log "pid $$ on $(hostname): ${#QUEUES[@]} queues: ${QUEUES[*]} (results $RESULTS)"

STOP=""
CHILD=""
on_signal() {
  STOP=$1
  log "received SIG$1: forwarding to the running queue (pid ${CHILD:-none}) and stopping"
  if [ -n "$CHILD" ]; then kill -"$1" "$CHILD" 2>/dev/null; fi
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap '' HUP

gpu_busy() {           # prints the compute apps holding the GPU; exit 2 when nvidia-smi fails
  local out
  if ! out=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1); then
    echo "nvidia-smi failed: $out"
    return 2
  fi
  printf '%s\n' "$out" | sed '/^[[:space:]]*$/d' | paste -sd ';' -
}

wait_gpu_free() {      # the previous queue's process may take a moment to release its CUDA context
  local waited=0 busy rc
  while :; do
    busy=$(gpu_busy); rc=$?
    if [ $rc -eq 2 ]; then
      if [ "${GSP_SKIP_GPU_GUARD:-0}" = "1" ]; then return 0; fi
      log "guard: $busy"; return 1
    fi
    [ -z "$busy" ] && return 0
    if [ "$waited" -ge "$GPU_WAIT_S" ]; then
      log "guard: the GPU is still busy after ${waited} s ($busy)"; return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
}

ANY_FAILED=0
i=0
for q in "${QUEUES[@]}"; do
  i=$((i + 1))
  if [ -n "$STOP" ]; then break; fi
  if ! wait_gpu_free; then
    log "stopping before queue $i ($q): guard failure (exit 3); re-launch when the GPU is free"
    exit 3
  fi
  log "queue $i/${#QUEUES[@]}: start $q"
  t0=$(date +%s)
  bash "$GSP_DIR/scripts/queue.sh" "$q" &
  CHILD=$!
  while :; do
    wait "$CHILD"; rc=$?
    if kill -0 "$CHILD" 2>/dev/null; then continue; fi      # wait was interrupted by a trapped signal
    break
  done
  CHILD=""
  log "queue $i/${#QUEUES[@]}: $q exited $rc after $(( $(date +%s) - t0 )) s"
  case $rc in
    0) ;;
    1) ANY_FAILED=1; log "queue $i finished with failed runs (retried on re-launch); continuing" ;;
    2) log "stopping: usage error from queue.sh"; exit 2 ;;
    3) log "stopping: guard failure (another queue process, or the GPU is busy)"; exit 3 ;;
    130|143) log "stopping: the queue was stopped by a signal (resumable)"; exit "$rc" ;;
    *) log "stopping: queue.sh exited $rc (see $LOG_DIR/queue_$(basename "$q" .jsonl).console.log)"; exit "$rc" ;;
  esac
done
if [ -n "$STOP" ]; then
  log "stopped by SIG$STOP before the end"
  [ "$STOP" = "INT" ] && exit 130 || exit 143
fi

log "gsp index"
"$PY" -m gsp.cli index --results "$RESULTS" || { log "gsp index failed"; exit 1; }
log "gsp aggregate"
"$PY" -m gsp.cli aggregate --results "$RESULTS" || { log "gsp aggregate failed"; exit 1; }
log "done (any failed runs: $ANY_FAILED)"
exit $ANY_FAILED
