#!/usr/bin/env bash
# Pull a machine's runs home and fold them into the local store (S6; PLAN §1.8: written, not used without approval).
#
#   bash GSP/scripts/remote/pull.sh TARGET [TAG]
#
# TARGET: user@host:/path (needs GSP_REMOTE_APPROVED=1, see common.sh) or a local directory; TAG names the shard
# (default: derived from TARGET). Steps:
#   1. rsync TARGET/GSP/results/runs/ -> results/incoming/TAG/runs/  (a mirror of the remote store; the local
#      results/runs is not written by rsync)
#   2. rsync TARGET/GSP/results/logs/ -> results/incoming/TAG/logs/  (queue logs, progress, per-run logs)
#   3. gsp merge --src results/incoming/TAG/runs: copies the remote's DONE runs into results/runs; never
#      overwrites a local done run (conflicts are reported); replaces local failed / running copies
#   4. gsp index (inside merge) -- then `gsp aggregate` and `gsp missing --queue Q` as usual.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ $# -lt 1 ]; then sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; fi
TARGET=${1%/}
TAG=${2:-$(printf '%s' "$TARGET" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//' | cut -c1-80)}
gsp_check_target "$TARGET"
IN="$RESULTS/incoming/$TAG"
mkdir -p "$IN/runs" "$IN/logs"
set -e
rsync "${RSYNC_OPTS[@]}" "$TARGET/GSP/results/runs/" "$IN/runs/"
rsync "${RSYNC_OPTS[@]}" "$TARGET/GSP/results/logs/" "$IN/logs/" || echo "(no remote logs)"
cd "$GSP_DIR"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m gsp.cli merge --src "$IN/runs" --results "$RESULTS"
