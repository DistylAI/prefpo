"""Tests for prefpo.__init__ — public API exports."""

import prefpo
def test_all_exports_importable():
    """Every name in __all__ is importable from prefpo."""
    for name in prefpo.__all__:
        obj = getattr(prefpo, name, None)
        assert obj is not None, f"{name} is in __all__ but not importable"
def test_all_exports_complete():
    """No public names missing from __all__."""
    expected = {
        "optimize",
        "optimize_async",
        "optimize_multi_trial",
        "PrefPOConfig",
        "ModelConfig",
        "DiscriminatorConfig",
        "OptimizerConfig",
        "OptimizationResult",
        "MultiTrialResult",
        "Prompt",
        "PromptRole",
        "Sample",
        "Grader",
        "GradeResult",
        "generate_outputs",
        "generate_standalone",
        "call_llm",
        "call_llm_json",
        "LLMResponse",
        "judge_prompt_hack",
        "judge_prompt_hygiene",
    }
    assert set(prefpo.__all__) == expected
