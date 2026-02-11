"""Tests for prefpo.ifeval_batch — compute_aggregate_metrics."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock
from prefpo.ifeval_batch import compute_aggregate_metrics
from prefpo.optimize import OptimizationResult
from prefpo.types import Prompt, PromptRole
def _make_result(score: float) -> OptimizationResult:
    """Create a minimal OptimizationResult for testing."""
    return OptimizationResult(
        run_id="run_test",
        best_prompt=Prompt(value="test", role=PromptRole.USER, name="test"),
        best_score=score,
        best_test_score=None,
        final_pool={"prompts": []},
        history=[],
        total_tokens={"discriminator": {}, "optimizer": {}},
    )
def test_compute_aggregate_metrics_empty():
    """Empty results returns total_samples=0."""
    result = compute_aggregate_metrics({})
    assert result["total_samples"] == 0
def test_compute_aggregate_metrics_single():
    """Single result computes correctly."""
    results = {0: _make_result(0.8)}
    metrics = compute_aggregate_metrics(results)
    assert metrics["total_samples"] == 1
    assert metrics["avg_final_score"] == 0.8
    assert metrics["std_final_score"] == 0.0
    assert metrics["min_score"] == 0.8
    assert metrics["max_score"] == 0.8
def test_compute_aggregate_metrics_multiple():
    """Mean, std, min, max computed correctly."""
    results = {
        0: _make_result(0.6),
        1: _make_result(0.8),
        2: _make_result(1.0),
    }
    metrics = compute_aggregate_metrics(results)
    assert metrics["total_samples"] == 3
    assert abs(metrics["avg_final_score"] - 0.8) < 0.01
    assert metrics["min_score"] == 0.6
    assert metrics["max_score"] == 1.0
    assert metrics["std_final_score"] > 0
def test_compute_aggregate_metrics_success_rate():
    """Counts score>=1.0 as success."""
    results = {
        0: _make_result(1.0),
        1: _make_result(0.5),
        2: _make_result(1.0),
        3: _make_result(0.0),
    }
    metrics = compute_aggregate_metrics(results)
    assert metrics["success_count"] == 2
    assert metrics["success_rate"] == 0.5
