#!/usr/bin/env bash
set -euo pipefail

SPACE_ONLY=0

if [[ "${1:-}" == "--space" ]]; then
  SPACE_ONLY=1
fi

python -m pip install --upgrade pip

if [[ "$SPACE_ONLY" -eq 0 ]]; then
  python -m pip install -r requirements-dev.txt
  python -m pip install -e .
fi

python -m pip install -r space/requirements.txt

cat <<'EOF'
PRISM-Memory setup complete.

Next steps:
  python space/app.py
  make test
EOF
