"""Test that RunLogger produces unique run IDs under rapid creation."""

import tempfile
from pathlib import Path
from prefpo.config import PrefPOConfig
from prefpo.results import RunLogger
def _make_config(tmpdir: str) -> PrefPOConfig:
    return PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"]},
        run={"iterations": 1, "output_dir": tmpdir},
    )
def test_single_run_no_collision():
    """Two single-run RunLoggers created back-to-back should get different dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger_a = RunLogger(config)
        logger_b = RunLogger(config)

        assert logger_a.run_id != logger_b.run_id, (
            f"Collision: both got {logger_a.run_id}"
        )
        assert logger_a.run_dir != logger_b.run_dir
def test_trial_index_no_collision():
    """Multi-trial RunLoggers with different trial indices should get different dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger_0 = RunLogger(config, trial_index=0)
        logger_1 = RunLogger(config, trial_index=1)

        assert logger_0.run_id != logger_1.run_id
        assert logger_0.run_dir != logger_1.run_dir
def test_many_single_runs_no_collision():
    """10 RunLoggers created in a tight loop should all be unique."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        loggers = [RunLogger(config) for _ in range(10)]
        run_ids = [l.run_id for l in loggers]

        assert len(set(run_ids)) == 10, (
            f"Collisions found: {run_ids}"
        )
