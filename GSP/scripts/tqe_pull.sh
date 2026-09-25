#!/usr/bin/env bash
# TQE rerun: mirror each remote worker's results_tqe (runs + logs) into results_tqe/incoming/<host>/ every PERIOD
# seconds, a backup while they run. Nothing local is overwritten; fold the runs in with
#   gsp merge --src results_tqe/incoming/<host>/runs --results results_tqe
#
#   bash GSP/scripts/tqe_pull.sh PERIOD HOST [HOST ...]      (HOST = ssh alias, e.g. vast_1 vast_2)
# The remote tree is ~/Quantum_Computing/GSP/results_tqe. Log: results_tqe/logs/tqe_pull.log.
set -u
GSP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES=${GSP_RESULTS:-$GSP_DIR/results_tqe}
if [ $# -lt 2 ]; then sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; fi
PERIOD=$1
shift
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ClearAllForwardings=yes"
mkdir -p "$RES/logs"
while true; do
  for h in "$@"; do
    mkdir -p "$RES/incoming/$h"
    ok=1
    for d in runs logs; do
      rsync -a --partial -e "$SSH" "$h:Quantum_Computing/GSP/results_tqe/$d/" "$RES/incoming/$h/$d/" >/dev/null 2>&1 || ok=0
    done
    n=$(find "$RES/incoming/$h/runs" -name run.json 2>/dev/null | wc -l)
    echo "[$(date -u +%FT%TZ)] $h: $([ $ok = 1 ] && echo pulled || echo 'pull FAILED') ($n run.json mirrored)"
  done >> "$RES/logs/tqe_pull.log"
  sleep "$PERIOD"
done
