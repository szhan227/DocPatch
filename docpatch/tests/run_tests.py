#!/usr/bin/env python
"""Minimal pytest-free test runner (this cluster has no network access to
install pytest). Discovers ``test_*.py`` in this directory, calls every
``test_*`` function, and injects the ``mixed_dev_dataset`` fixture by name
for functions that declare it as a parameter.

Usage: PYTHONPATH=src:. python docpatch/tests/run_tests.py
"""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
MIXED_DEV_PATH = REPO_ROOT / "data_preprocess" / "outputs" / "mixed" / "docpatch_dev.jsonl"


def build_fixtures() -> dict:
    fixtures = {}
    if MIXED_DEV_PATH.exists():
        from ctx_to_lora.docpatch.data import DocPatchDataset

        fixtures["mixed_dev_dataset"] = DocPatchDataset(MIXED_DEV_PATH)
    return fixtures


class Skipped(Exception):
    pass


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    fixtures = build_fixtures()

    passed, failed, skipped = 0, 0, 0
    failures = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f"docpatch.tests.{path.stem}"
        module = importlib.import_module(module_name)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            params = list(inspect.signature(func).parameters)
            kwargs = {}
            missing_fixture = False
            for p in params:
                if p in fixtures:
                    kwargs[p] = fixtures[p]
                else:
                    missing_fixture = True
            try:
                if missing_fixture:
                    raise Skipped(f"missing fixture for one of {params}")
                func(**kwargs)
                passed += 1
                print(f"PASS {path.stem}::{name}")
            except Skipped as exc:
                skipped += 1
                print(f"SKIP {path.stem}::{name} ({exc})")
            except Exception:
                failed += 1
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
                failures.append(f"{path.stem}::{name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("Failed tests:", ", ".join(failures))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
