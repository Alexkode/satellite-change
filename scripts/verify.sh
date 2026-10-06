#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python -m pytest tests -q
python scripts/check_evidence.py
