from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_missing_comparison_experiments.py"


def test_missing_comparison_script_avoids_slow_vector_imports() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for forbidden in [
        "epfl_random_vector_helpers",
        "materialize_reviewer_experiments",
        "run_epfl20_vector_stability",
    ]:
        assert forbidden not in source


def test_missing_comparison_script_labels_scoap_as_proxy() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "scoap_proxy" in source
    assert "not external SCOAP" in source
    assert "target FI labels before ranking" in source
    assert "target oracle counts before ranking" in source
    assert "target random-vector outcomes before ranking" in source
