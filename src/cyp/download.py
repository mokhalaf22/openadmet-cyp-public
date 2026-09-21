"""Download the challenge dataset at a pinned revision.

The revision is pinned so every run — and every reader of the report — gets the
exact same bytes. If the upstream dataset is updated, bump ``DATASET_REVISION``
here and in CLAUDE.md in the same commit, and re-inspect.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_REPO = "openadmet/cyp-challenge-train-test"
# Pinned revision — see CLAUDE.md "Dataset". Do not download HEAD.
DATASET_REVISION = "3ac9c5dbb83eec5780ec7fa511908698cfe1396d"
LOCAL_DIR = Path("data") / "cyp-challenge-train-test"


def download() -> Path:
    path = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(LOCAL_DIR),
    )
    return Path(path)


def main() -> int:
    path = download()
    print(f"Downloaded {DATASET_REPO}@{DATASET_REVISION[:12]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
