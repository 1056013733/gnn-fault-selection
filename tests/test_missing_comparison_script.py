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


def test_missing_comparison_script_keeps_observability_proxy_label() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "observability_cone_proxy" in source
    assert "co_only_observability" in source
    assert "external tool measurements" in source
    forbidden = "sco" + "ap"
    assert forbidden not in source.lower()
    assert "target FI labels before ranking" in source
    assert "target oracle counts before ranking" in source
    assert "target random-vector outcomes before ranking" in source
