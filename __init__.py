"""PrefPO — Preference-based Prompt Optimization."""

from prefpo.config import (
    DiscriminatorConfig,
    ModelConfig,
    OptimizerConfig,
    PrefPOConfig,
)
from prefpo.generate import generate_outputs, generate_standalone
from prefpo.grading.base import GradeResult, Grader
from prefpo.judges.hack import judge_prompt_hack
from prefpo.judges.hygiene import judge_prompt_hygiene
from prefpo.llm.client import LLMResponse, call_llm, call_llm_json
from prefpo.optimize import (
    MultiTrialResult,
    OptimizationResult,
    optimize,
    optimize_async,
    optimize_multi_trial,
)
from prefpo.types import Prompt, PromptRole, Sample

__all__ = [
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
]
