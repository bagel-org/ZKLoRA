import json
from pathlib import Path
import sys
from pathlib import Path as _P

import pytest
np = pytest.importorskip("numpy")

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
from zklora import polynomial_commit
commit_activations = polynomial_commit.commit_activations
verify_commitment = polynomial_commit.verify_commitment


def test_commit_roundtrip(tmp_path: Path):
    data = {"input_data": [1, 2, 3, 4]}
    path = tmp_path / "acts.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)
