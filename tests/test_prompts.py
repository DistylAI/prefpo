"""Tests for prefpo.prompts — discriminator and optimizer prompt builders."""

from prefpo.config import DiscriminatorConfig, OptimizerConfig
from prefpo.prompts.discriminator import (
    DISCRIMINATOR_SCHEMA,
    build_discriminator_prompt,
    build_instruction_trajectory,
)
from prefpo.prompts.optimizer import OPTIMIZER_SCHEMA, build_optimizer_prompt
from prefpo.types import ModelOutput, Prompt, PromptRole, Sample
# --- Trajectory tests ---
def test_instruction_trajectory_with_target():
    outputs = [ModelOutput(sample_index=0, prompt_sent="p", response="The answer is A")]
    samples = [Sample(index=0, question="Pick one", target="A")]
    traj = build_instruction_trajectory(outputs, samples, show_expected=True)
    assert "Question:" in traj
    assert "Response:" in traj
    assert "Expected Answer:" in traj
    assert "A" in traj
def test_instruction_trajectory_without_target():
    outputs = [ModelOutput(sample_index=0, prompt_sent="p", response="The answer is A")]
    samples = [Sample(index=0, question="Pick one", target="A")]
    traj = build_instruction_trajectory(outputs, samples, show_expected=False)
    assert "Expected Answer:" not in traj
def test_instruction_trajectory_multiple_samples():
    outputs = [
        ModelOutput(sample_index=0, prompt_sent="p", response="resp1"),
        ModelOutput(sample_index=1, prompt_sent="p", response="resp2"),
    ]
    samples = [
        Sample(index=0, question="Q1", target="A"),
        Sample(index=1, question="Q2", target="B"),
    ]
    traj = build_instruction_trajectory(outputs, samples, show_expected=True)
    assert "Sample 1" in traj
    assert "Sample 2" in traj
    assert "resp1" in traj
    assert "resp2" in traj
# --- Discriminator prompt tests ---
def test_discriminator_prompt_basic():
    cfg = DiscriminatorConfig()
    sys_p, user_p = build_discriminator_prompt("traj_a", "traj_b", cfg)
    assert "evaluator" in sys_p.lower()
    assert "<Version 1>" in user_p
    assert "<Version 2>" in user_p
    assert "traj_a" in user_p
    assert "traj_b" in user_p
def test_discriminator_prompt_criteria_string():
    cfg = DiscriminatorConfig(criteria="correctness")
    _, user_p = build_discriminator_prompt("a", "b", cfg)
    assert "CRITERIA TO EVALUATE ON:" in user_p
    assert "- correctness" in user_p
def test_discriminator_prompt_criteria_list():
    cfg = DiscriminatorConfig(criteria=["accuracy", "reasoning quality"])
    _, user_p = build_discriminator_prompt("a", "b", cfg)
    assert "- accuracy" in user_p
    assert "- reasoning quality" in user_p
def test_discriminator_prompt_constraints():
    cfg = DiscriminatorConfig(constraints=["be concise", "no jargon"])
    _, user_p = build_discriminator_prompt("a", "b", cfg)
    assert "ADDITIONAL INFORMATION:" in user_p
    assert "- be concise" in user_p
    assert "- no jargon" in user_p
def test_discriminator_prompt_no_criteria_no_block():
    cfg = DiscriminatorConfig(criteria="", constraints="")
    _, user_p = build_discriminator_prompt("a", "b", cfg)
    assert "CRITERIA TO EVALUATE ON:" not in user_p
    assert "ADDITIONAL INFORMATION:" not in user_p
def test_discriminator_schema_structure():
    assert DISCRIMINATOR_SCHEMA["type"] == "json_schema"
    assert DISCRIMINATOR_SCHEMA["schema"]["properties"]["preferred"]["enum"] == [1, 2]
    assert DISCRIMINATOR_SCHEMA["schema"]["additionalProperties"] is False
# --- Optimizer prompt tests ---
def test_optimizer_prompt_preferred_1():
    p = Prompt(value="bad instruction", role=PromptRole.USER)
    cfg = OptimizerConfig()
    prompt = build_optimizer_prompt(preferred=1, non_preferred_prompt=p, feedback="needs work", config=cfg)
    assert "Version 2 Instruction: bad instruction" in prompt
    assert "needs work" in prompt
def test_optimizer_prompt_preferred_2():
    p = Prompt(value="bad instruction", role=PromptRole.USER)
    cfg = OptimizerConfig()
    prompt = build_optimizer_prompt(preferred=2, non_preferred_prompt=p, feedback="needs work", config=cfg)
    assert "Version 1 Instruction: bad instruction" in prompt
def test_optimizer_prompt_constraints():
    p = Prompt(value="x", role=PromptRole.USER)
    cfg = OptimizerConfig(constraints="keep format rules")
    prompt = build_optimizer_prompt(preferred=1, non_preferred_prompt=p, feedback="f", config=cfg)
    assert "CONSTRAINTS FOR YOUR OUTPUT:" in prompt
    assert "- keep format rules" in prompt
def test_optimizer_prompt_no_constraints():
    p = Prompt(value="x", role=PromptRole.USER)
    cfg = OptimizerConfig(constraints="")
    prompt = build_optimizer_prompt(preferred=1, non_preferred_prompt=p, feedback="f", config=cfg)
    assert "CONSTRAINTS FOR YOUR OUTPUT:" not in prompt
def test_optimizer_schema_structure():
    assert OPTIMIZER_SCHEMA["type"] == "json_schema"
    assert "prompt" in OPTIMIZER_SCHEMA["schema"]["properties"]
    assert OPTIMIZER_SCHEMA["schema"]["additionalProperties"] is False
# --- Standalone trajectory tests ---

from prefpo.prompts.discriminator import (
    _format_criteria_block,
    _format_constraints_block,
    build_standalone_trajectory,
)
def test_standalone_trajectory_basic():
    """build_standalone_trajectory without show_expected."""
    outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="hello")]

    class _Grader:
        def check_output(self, output, prompt_text=None):
            return {"ok": True}

    traj = build_standalone_trajectory(outputs, _Grader(), show_expected=False)
    assert "Output:" in traj
    assert "hello" in traj
    assert "Grade:" not in traj
def test_standalone_trajectory_with_grade():
    """build_standalone_trajectory with show_expected and grader."""
    outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="hello")]

    class _Grader:
        def check_output(self, output, prompt_text=None):
            return {"passed": True, "score": 1.0}

    traj = build_standalone_trajectory(outputs, _Grader(), show_expected=True, prompt_text="test")
    assert "Grade:" in traj
    assert "passed" in traj
def test_standalone_trajectory_none_check_output_raises():
    """Raises ValueError when check_output returns None."""
    outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="hello")]

    class _Grader:
        def check_output(self, output, prompt_text=None):
            return None

    import pytest
    with pytest.raises(ValueError, match="check_output"):
        build_standalone_trajectory(outputs, _Grader(), show_expected=True, prompt_text="test")
# --- Edge cases for prompt building ---
def test_optimizer_prompt_empty_feedback():
    """Empty string feedback is handled."""
    p = Prompt(value="instruction", role=PromptRole.USER)
    cfg = OptimizerConfig()
    prompt = build_optimizer_prompt(preferred=1, non_preferred_prompt=p, feedback="", config=cfg)
    assert "Feedback:" in prompt
def test_optimizer_prompt_empty_prompt_value():
    """prompt.value="" is passed through."""
    p = Prompt(value="", role=PromptRole.USER)
    cfg = OptimizerConfig()
    prompt = build_optimizer_prompt(preferred=1, non_preferred_prompt=p, feedback="fix it", config=cfg)
    assert "Version 2 Instruction: " in prompt
def test_format_criteria_block_empty_list():
    """Empty list returns empty string."""
    assert _format_criteria_block([]) == ""
    assert _format_criteria_block("") == ""
def test_format_constraints_block_single_string():
    """Single string becomes one bullet."""
    result = _format_constraints_block("be concise")
    assert "- be concise" in result
    assert "ADDITIONAL INFORMATION:" in result
