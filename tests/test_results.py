"""Tests for prefpo.results — RunLogger."""

import json
import re
import tempfile
from pathlib import Path
from prefpo.config import PrefPOConfig
from prefpo.results import RunLogger
from prefpo.types import IterationRecord, Prompt, PromptRole
def _make_config(tmpdir: str) -> PrefPOConfig:
    return PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"]},
        run={"iterations": 1, "output_dir": tmpdir},
    )
def test_run_dir_created():
    """RunLogger creates a directory on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        assert logger.run_dir.exists()
        assert logger.run_dir.is_dir()
def test_config_json_written():
    """config.json is written on init with correct content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        config_path = logger.run_dir / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["mode"] == "instruction"
        assert data["task_model"]["name"] == "openai/gpt-4o"
def test_run_id_format():
    """run_id matches run_YYYYMMDD_HHMMSS_HEXHEX pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        assert re.match(r"^run_\d{8}_\d{6}_[0-9a-f]{6}$", logger.run_id)
def test_run_id_with_trial_index():
    """trial_index is included in run_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config, trial_index=3)
        assert re.match(r"^run_\d{8}_\d{6}_3_[0-9a-f]{6}$", logger.run_id)
def _make_record(iteration: int = 0) -> IterationRecord:
    pa = Prompt(value="a", role=PromptRole.USER, name="a")
    pb = Prompt(value="b", role=PromptRole.USER, name="b")
    imp = Prompt(value="c", role=PromptRole.USER, name="improved")
    return IterationRecord(
        iteration=iteration,
        prompt_a=pa,
        prompt_b=pb,
        prompt_a_score=0.5,
        prompt_b_score=0.6,
        preferred=1,
        feedback="good",
        improved_prompt=imp,
        improved_score=0.7,
        discriminator_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        optimizer_usage={"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
    )
def test_log_iteration_creates_jsonl():
    """log_iteration creates iteration_history.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        logger.log_iteration(_make_record())
        history_path = logger.run_dir / "iteration_history.jsonl"
        assert history_path.exists()
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["iteration"] == 0
        assert data["preferred"] == 1
def test_log_iteration_appends():
    """Multiple log_iteration calls append lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        logger.log_iteration(_make_record(0))
        logger.log_iteration(_make_record(1))
        logger.log_iteration(_make_record(2))
        history_path = logger.run_dir / "iteration_history.jsonl"
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[2])["iteration"] == 2
def test_log_final_pool():
    """log_final_pool writes final_pool_state.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        pool_state = {"prompts": [{"name": "seed_0", "value": "test", "score": 0.8}]}
        logger.log_final_pool(pool_state)
        path = logger.run_dir / "final_pool_state.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["prompts"][0]["score"] == 0.8
def test_log_summary():
    """log_summary writes summary.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        logger = RunLogger(config)
        summary = {"run_id": logger.run_id, "best_val_score": 0.9}
        logger.log_summary(summary)
        path = logger.run_dir / "summary.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["best_val_score"] == 0.9
def test_custom_base_dir():
    """base_dir overrides config.run.output_dir."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        config = _make_config(tmpdir1)
        logger = RunLogger(config, base_dir=Path(tmpdir2))
        assert str(logger.run_dir).startswith(tmpdir2)
        assert not str(logger.run_dir).startswith(tmpdir1)
