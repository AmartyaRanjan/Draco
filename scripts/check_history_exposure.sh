#!/usr/bin/env bash
# Read-only check: are the removed runtime files still reachable from Git
# history (DRACO-010)? Exit 1 while any blob is reachable, 0 once purged.
# Safe to run in any clone; see docs/HISTORY_PURGE.md for remediation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# rev-list separates the object name from its path with a single space.
pattern='(^| )app/storage/(documents\.db|uploads/)|(^| )eval_results/'
found="$(git rev-list --all --objects | grep -E "$pattern" || true)"

if [ -n "$found" ]; then
    echo "Runtime data is STILL REACHABLE from Git history:"
    echo "$found"
    echo "Remediation steps: docs/HISTORY_PURGE.md"
    exit 1
fi

echo "Clean: no runtime-data blobs are reachable from any ref."
