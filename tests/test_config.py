"""Tests for prefpo.config — Pydantic config models."""

import tempfile
from pathlib import Path

import pytest
from prefpo.config import (
    DiscriminatorConfig,
    ModelConfig,
    OptimizerConfig,
    PoolConfig,
    PrefPOConfig,
    RunConfig,
)
def test_model_config_defaults():
    mc = ModelConfig(name="openai/gpt-4o")
    assert mc.is_reasoning is False
    assert mc.reasoning_effort == "medium"
    assert mc.temperature == 0.0
def test_model_config_reasoning():
    mc = ModelConfig(name="openai/gpt-5", is_reasoning=True, reasoning_effort="high")
    assert mc.is_reasoning is True
    assert mc.reasoning_effort == "high"
def test_discriminator_config_defaults():
    dc = DiscriminatorConfig()
    assert dc.model.name == "openai/gpt-5"
    assert dc.model.is_reasoning is True
    assert dc.criteria == ""
    assert dc.constraints == ""
    assert dc.show_expected is False
def test_discriminator_config_list_criteria():
    dc = DiscriminatorConfig(criteria=["accuracy", "reasoning"])
    assert dc.criteria == ["accuracy", "reasoning"]
def test_optimizer_config_defaults():
    oc = OptimizerConfig()
    assert oc.model.name == "openai/gpt-5"
    assert oc.constraints == ""
def test_pool_config_validation():
    pc = PoolConfig(initial_prompts=["hello"])
    assert pc.prompt_role == "user"
    assert pc.sampling_seed == 42
def test_pool_config_empty_prompts_rejected():
    with pytest.raises(Exception):
        PoolConfig(initial_prompts=[])
def test_run_config_defaults():
    rc = RunConfig()
    assert rc.iterations == 5
    assert rc.n_trials == 1
    assert rc.vary_seed is False
    assert rc.max_concurrent == 100
def test_prefpo_config_from_dict():
    cfg = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"]},
    )
    assert cfg.mode == "instruction"
    assert cfg.task_model.name == "openai/gpt-4o"
    assert cfg.discriminator.model.name == "openai/gpt-5"
def test_prefpo_config_from_yaml():
    yaml_content = """
mode: "instruction"
task_model:
  name: "openai/gpt-4o"
  temperature: 0.0
pool:
  initial_prompts:
    - "test prompt"
  prompt_role: "user"
run:
  iterations: 3
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = PrefPOConfig.from_yaml(f.name)

    assert cfg.mode == "instruction"
    assert cfg.task_model.name == "openai/gpt-4o"
    assert cfg.run.iterations == 3
    assert cfg.pool.initial_prompts == ["test prompt"]
def test_invalid_mode_rejected():
    with pytest.raises(Exception):
        PrefPOConfig(
            mode="invalid",
            task_model={"name": "openai/gpt-4o"},
            pool={"initial_prompts": ["test"]},
        )
def test_pool_config_update_strategy():
    pc = PoolConfig(initial_prompts=["a"], update_strategy="replace")
    assert pc.update_strategy == "replace"

    with pytest.raises(Exception):
        PoolConfig(initial_prompts=["a"], update_strategy="invalid")
# --- Error path tests ---
def test_from_yaml_file_not_found():
    """Raises FileNotFoundError for nonexistent YAML."""
    with pytest.raises(FileNotFoundError):
        PrefPOConfig.from_yaml("/tmp/nonexistent_config_12345.yaml")
def test_from_yaml_invalid_content():
    """Raises validation error on YAML with wrong types."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("mode: 999\ntask_model: not_a_dict\n")
        f.flush()
        with pytest.raises(Exception):
            PrefPOConfig.from_yaml(f.name)
def test_invalid_reasoning_effort():
    """Rejected by Literal validation."""
    with pytest.raises(Exception):
        ModelConfig(name="openai/gpt-4o", reasoning_effort="extreme")
def test_invalid_prompt_role():
    """Rejected by Literal validation."""
    with pytest.raises(Exception):
        PoolConfig(initial_prompts=["a"], prompt_role="assistant")
def test_config_model_copy_deep():
    """model_copy(deep=True) creates independent copy."""
    cfg = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"]},
    )
    copy = cfg.model_copy(deep=True)
    copy.run.iterations = 99
    assert cfg.run.iterations != 99
