#!/usr/bin/env bash
# Push the harness to a machine that will run a queue (S6; PLAN §1.8: written, not used without approval).
#
#   bash GSP/scripts/remote/push.sh TARGET [QUEUE.jsonl ...]
#
# TARGET: user@host:/path (needs GSP_REMOTE_APPROVED=1, see common.sh) or a local directory. Copies into TARGET/GSP/:
#   the code (gsp/, configs/, env/, scripts/, tests/, pyproject.toml, docs) without results/ and caches;
#   results/instances/ (the frozen instances, seed table, CHECKSUMS) and results/sectors/ (the sector files);
#   the given queue files (+ their .plan.json) into results/queues/.
# Nothing is deleted on the target (no --delete), so a remote results/runs is never touched.
# On the remote host afterwards (manual, with approval): create the `gsp` env from env/README.md
# (pip install -r env/requirements.lock; pip install -e GSP --no-deps), then
#   nohup bash GSP/scripts/queue.sh GSP/results/queues/NAME.jsonl >/dev/null 2>&1 &
# and bring the runs home with pull.sh.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ $# -lt 1 ]; then sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; fi
TARGET=${1%/}
shift
gsp_check_target "$TARGET"
if ! gsp_is_remote "$TARGET"; then mkdir -p "$TARGET/GSP/results/queues"; fi
set -e
rsync "${RSYNC_OPTS[@]}" --exclude '/results/' --exclude '__pycache__/' --exclude '*.egg-info/' \
      --exclude '.pytest_cache/' --exclude '.venv/' "$GSP_DIR/" "$TARGET/GSP/"
rsync "${RSYNC_OPTS[@]}" "$RESULTS/instances" "$RESULTS/sectors" "$TARGET/GSP/results/"
for q in "$@"; do
  side="${q%.jsonl}.plan.json"
  files=("$q")
  [ -f "$side" ] && files+=("$side")
  rsync "${RSYNC_OPTS[@]}" "${files[@]}" "$TARGET/GSP/results/queues/"
done
echo "pushed code + instances + sectors$([ $# -gt 0 ] && echo " + $# queue(s)") to $TARGET/GSP"
