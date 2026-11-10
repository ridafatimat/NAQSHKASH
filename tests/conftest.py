"""
Pytest configuration for preprocessing tests.
Adds the repository root directory to sys.path.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional integration checks need the externally distributed sample datasets.
import inspect
import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        dataset = getattr(item.module, 'DATASET_PATH', None)
        if dataset is not None and not dataset.exists():
            source = inspect.getsource(item.obj)
            if 'DATASET_PATH' in source or 'LABELS_PATH' in source:
                item.add_marker(pytest.mark.skip(reason=f'External sample dataset not installed: {dataset}'))
