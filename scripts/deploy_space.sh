#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/dist/space_bundle"
SPACE_REPO="${1:-}"

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/results" "$BUNDLE_DIR/docs/release"

cp "$ROOT_DIR/space/app.py" "$BUNDLE_DIR/app.py"
cp "$ROOT_DIR/space/README.md" "$BUNDLE_DIR/README.md"
cp "$ROOT_DIR/space/requirements.txt" "$BUNDLE_DIR/requirements.txt"
cp "$ROOT_DIR/docs/release/extraction-skill.md" "$BUNDLE_DIR/docs/release/extraction-skill.md"
cp "$ROOT_DIR/docs/release/datasets.md" "$BUNDLE_DIR/docs/release/datasets.md"
cp "$ROOT_DIR/docs/release/extraction-examples.md" "$BUNDLE_DIR/docs/release/extraction-examples.md"
cp "$ROOT_DIR/docs/release/release-results.md" "$BUNDLE_DIR/docs/release/release-results.md"
cp "$ROOT_DIR/results/confirmed_exp15_summary.json" "$BUNDLE_DIR/results/confirmed_exp15_summary.json"
cp "$ROOT_DIR/results/readme_extraction_examples.json" "$BUNDLE_DIR/results/readme_extraction_examples.json"
cp "$ROOT_DIR/results/scenario_comparisons.json" "$BUNDLE_DIR/results/scenario_comparisons.json"

echo "Prepared Space bundle at: $BUNDLE_DIR"

if [[ -z "$SPACE_REPO" ]]; then
  exit 0
fi

python - "$SPACE_REPO" "$BUNDLE_DIR" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required to upload the Space. "
        "Install it with: python -m pip install huggingface_hub"
    ) from exc

repo_id = sys.argv[1]
bundle_dir = Path(sys.argv[2])

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=str(bundle_dir),
    commit_message="Update PRISM-Memory Space bundle",
)
print(f"Uploaded bundle to https://huggingface.co/spaces/{repo_id}")
PY
