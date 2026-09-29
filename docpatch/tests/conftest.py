from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIXED_DEV_PATH = REPO_ROOT / "data_preprocess" / "outputs" / "mixed" / "docpatch_dev.jsonl"


@pytest.fixture(scope="session")
def mixed_dev_dataset():
    if not MIXED_DEV_PATH.exists():
        pytest.skip(f"mixed dev dataset not found at {MIXED_DEV_PATH}")
    from ctx_to_lora.docpatch.data import DocPatchDataset

    return DocPatchDataset(MIXED_DEV_PATH)
