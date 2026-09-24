#!/usr/bin/env bash
# Shared by push.sh / pull.sh (S6). Sourced, not run.
#
# A TARGET is either user@host:/path (a remote host, over ssh) or a plain local directory (tests, dry runs).
# PLAN §1.8: no GPU rental and no remote run without Sensei's explicit approval, so a remote target is refused
# unless GSP_REMOTE_APPROVED=1 is set in the environment by whoever has that approval. Written in S6, tested only
# against local directories.
#   GSP_REMOTE_SSH   ssh command for rsync -e (default: "ssh"), e.g. "ssh -p 2222 -i ~/.ssh/vast"
#   GSP_RESULTS      local results root (default GSP/results);  GSP_PY  python of the gsp env
set -u
PY=${GSP_PY:-$HOME/anaconda3/envs/gsp/bin/python}
GSP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS=${GSP_RESULTS:-$GSP_DIR/results}
RSYNC_OPTS=(-a --partial)

gsp_is_remote() {   # user@host:/path or host:path (a local path never contains ':' here)
  [[ "$1" == *:* ]]
}

gsp_check_target() {
  local t=$1
  if gsp_is_remote "$t"; then
    if [ "${GSP_REMOTE_APPROVED:-0}" != "1" ]; then
      echo "refused: $t is a remote host. PLAN §1.8: no remote run without Sensei's explicit approval;" \
           "set GSP_REMOTE_APPROVED=1 only with it." >&2
      exit 4
    fi
    RSYNC_OPTS+=(-e "${GSP_REMOTE_SSH:-ssh}" --compress)
  fi
}
