#!/usr/bin/env bash
# Self-healing supervisor for the LATE data ingest.
#
# Re-runs the resumable ingest until a full pass adds zero new files (converged),
# repairing any corrupt/partial parquet files between passes. Safe to run detached
# (nohup): if the Python process dies for any reason, the loop restarts it, and
# skip-if-exists means no already-downloaded data is re-fetched.
set -u

cd /home/dennis/Projects/Basketball || exit 1
LOG=ingest-full.log
PY=.venv/bin/python
MAX_PASSES=40

count_files() { find data/raw -name '*.parquet' 2>/dev/null | wc -l; }

for pass in $(seq 1 "$MAX_PASSES"); do
  # Repair: drop empty parquets and stray temp files left by any prior crash/kill.
  find data/raw -name '*.parquet' -size 0 -delete 2>/dev/null
  find data/raw -name '*.parquet.tmp' -delete 2>/dev/null

  before=$(count_files)
  echo "$(date '+%F %T') === supervisor pass $pass start (files=$before) ===" >> "$LOG"

  "$PY" -m elevate_stat.run_ingest >> "$LOG" 2>&1
  code=$?

  after=$(count_files)
  echo "$(date '+%F %T') === pass $pass end (code=$code, files ${before}->${after}) ===" >> "$LOG"

  # Converged: a clean pass that produced nothing new (everything fetchable is fetched).
  if [ "$code" -eq 0 ] && [ "$before" -eq "$after" ]; then
    echo "$(date '+%F %T') === CONVERGED after pass $pass (files=$after) ===" >> "$LOG"
    break
  fi

  sleep 30  # brief cooldown before the next pass (eases any throttling)
done

echo "$(date '+%F %T') === supervisor exiting ===" >> "$LOG"
