#!/usr/bin/env bash
# Four arms, identical data/sampler/seed/schedule. Run OUTSIDE a sandbox, in the foreground.
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-30}"
for mode in off modality_only quality_only full; do
  name="unified_${mode}"
  [ -f "outputs/runs/${name}/metrics.json" ] && { echo "skip ${name} (done)"; continue; }
  rm -rf "outputs/runs/${name}"
  ../.venv/bin/python scripts/train.py --name "${name}" --mode "${mode}" --epochs "${EPOCHS}"
  ../.venv/bin/python scripts/evaluate.py --name "${name}"
done
