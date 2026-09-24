#!/usr/bin/env bash
# Unattended queue chain (S9c): waits for a running s9_calibration.sh, resumes the S9a queues, then runs the queues
# listed in a chain file one at a time, each through scripts/s9_calibration.sh (its lock, GPU guard, logs, and the
# closing `gsp index` + `gsp aggregate`). One GPU process at a time throughout.
#
#   setsid nohup bash GSP/scripts/chain.sh GSP/results/queues/s9c.chain >/dev/null 2>&1 </dev/null &
#
# The chain file: one queue file per line (absolute, or relative to the results root); blank lines and '#' comments
# are skipped. It is re-read before every queue, so lines may be appended while the chain runs. Each queue runs once
# per launch; a re-launch resumes (done run_ids are skipped, failed / interrupted runs are retried).
# Before the first chain queue it runs s9_calibration.sh with its default three S9a queues (resume; quick when done).
#
# A queue whose runs fail does not stop the chain. A guard refusal (exit 3: the GPU or a lock is busy) is retried
# every CHAIN_RETRY_S (600) up to CHAIN_RETRIES (6) times, then the queue is skipped. SIGTERM / SIGINT are forwarded
# to the running launcher and stop the chain. Log: results/logs/chain.log. Stop: kill -TERM "$(cat .../chain.pid)".
set -u
GSP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS=${GSP_RESULTS:-$GSP_DIR/results}
LOG_DIR="$RESULTS/logs"
RETRY_S=${CHAIN_RETRY_S:-600}
RETRIES=${CHAIN_RETRIES:-6}
mkdir -p "$LOG_DIR"
exec </dev/null >>"$LOG_DIR/chain.log" 2>&1

log() { echo "[$(date -u +%FT%TZ)] chain: $*"; }

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
  log "usage: chain.sh CHAIN_FILE (got: $*)"
  exit 2
fi
CHAIN=$(readlink -f "$1")
exec 9>>"$LOG_DIR/chain.lock"
if ! flock -n 9; then
  log "refused: another chain.sh holds $LOG_DIR/chain.lock"
  exit 3
fi
echo $$ > "$LOG_DIR/chain.pid"
log "pid $$: chain file $CHAIN (results $RESULTS)"

STOP=""
CHILD=""
on_signal() {
  STOP=$1
  log "received SIG$1: forwarding to pid ${CHILD:-none} and stopping"
  if [ -n "$CHILD" ]; then kill -"$1" "$CHILD" 2>/dev/null; fi
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap '' HUP

launch() {             # launch LABEL [queue files ...]: s9_calibration.sh on the given queues (none = its defaults)
  local label=$1 rc tries=0
  shift
  while :; do
    [ -n "$STOP" ] && return 143
    log "$label: start ${*:-(the S9a queues)}"
    if [ $# -gt 0 ]; then
      S9_QUEUES="$*" bash "$GSP_DIR/scripts/s9_calibration.sh" &
    else
      env -u S9_QUEUES bash "$GSP_DIR/scripts/s9_calibration.sh" &
    fi
    CHILD=$!
    while :; do
      wait "$CHILD"; rc=$?
      if kill -0 "$CHILD" 2>/dev/null; then continue; fi
      break
    done
    CHILD=""
    log "$label: exited $rc"
    if [ "$rc" -eq 3 ] && [ -z "$STOP" ] && [ "$tries" -lt "$RETRIES" ]; then
      tries=$((tries + 1))
      log "$label: guard refusal; retry $tries/$RETRIES in $RETRY_S s"
      sleep "$RETRY_S" & CHILD=$!; wait "$CHILD"; CHILD=""
      continue
    fi
    return "$rc"
  done
}

# 1. wait for a running s9_calibration.sh (blocks on its lock, releases it at once)
log "waiting for $LOG_DIR/s9_calibration.lock"
flock "$LOG_DIR/s9_calibration.lock" true &
CHILD=$!; wait "$CHILD"; CHILD=""
[ -n "$STOP" ] && exit 143
log "s9_calibration lock is free"
sleep 20                 # let the previous process release the GPU

# 2. resume the S9a queues (skips every done run)
launch "resume S9a"

# 3. the chain file, re-read before every queue
DONE=()
while [ -z "$STOP" ]; do
  NEXT=""
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; line="$(echo "$line" | xargs)"
    [ -z "$line" ] && continue
    case $line in /*) q=$line ;; *) q="$RESULTS/$line" ;; esac
    seen=0
    for d in "${DONE[@]+"${DONE[@]}"}"; do [ "$d" = "$q" ] && seen=1 && break; done
    if [ $seen -eq 0 ]; then NEXT=$q; break; fi
  done < "$CHAIN"
  [ -z "$NEXT" ] && break
  DONE+=("$NEXT")
  if [ ! -f "$NEXT" ]; then
    log "skip: no such queue file: $NEXT"
    continue
  fi
  launch "$(basename "$NEXT" .jsonl)" "$NEXT"
done
if [ -n "$STOP" ]; then
  log "stopped by SIG$STOP"
  exit 143
fi
log "done: ${#DONE[@]} chain queues attempted"
exit 0
