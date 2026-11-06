#!/usr/bin/env bash
# Verify every shard in DIR that has candidate periods and no *_verified.json yet; repeat until none left.
DIR=${1:?usage: verify_all.sh SHARDS_DIR}
PY=${PY:-$HOME/.venvs/nnperiods/bin/python}
cd "$(dirname "$0")"
while true; do
    todo=$(for m in "$DIR"/*_meta.json; do s=$(basename "$m" _meta.json); [ -e "$DIR/${s}_verified.json" ] || [ "$s" = vvv_quiet ] || echo "$s"; done)
    [ -z "$todo" ] && { sleep 600; continue; }
    for s in $todo; do echo "== $s $(date +%T)"; "$PY" verify_periods.py --data "$DIR" --shard "$s" 2>&1 | grep -E "^done|Error|Traceback"; done
done
