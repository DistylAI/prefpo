"""Pydantic configuration models for PrefPO."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    """Config for any model call (task model, discriminator, or optimizer)."""

    name: str
    is_reasoning: bool = False
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    temperature: float = 0.0


class DiscriminatorConfig(BaseModel):
    """Config for the discriminator (judge) model."""

    model: ModelConfig = ModelConfig(name="openai/gpt-5", is_reasoning=True)
    criteria: str | list[str] = ""
    constraints: str | list[str] = ""
    show_expected: bool = False


class OptimizerConfig(BaseModel):
    """Config for the optimizer model."""

    model: ModelConfig = ModelConfig(name="openai/gpt-5", is_reasoning=True)
    constraints: str | list[str] = ""


class PoolConfig(BaseModel):
    """Config for the prompt pool."""

    initial_prompts: list[str]
    prompt_role: Literal["user", "system"] = "user"
    update_strategy: Literal["add", "replace"] = "add"
    sampling_seed: int = 42

    @field_validator("initial_prompts")
    @classmethod
    def validate_initial_prompts(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("initial_prompts must contain at least one prompt")
        return v


class RunConfig(BaseModel):
    """Config for optimization run settings."""

    iterations: int = 5
    n_trials: int = 1
    vary_seed: bool = False
    max_concurrent: int = 100
    output_dir: str = "results/prefpo"
    save_outputs: bool = False
    verbose: bool = True


class PrefPOConfig(BaseModel):
    """Top-level configuration for a PrefPO optimization run."""

    mode: Literal["instruction", "standalone"]
    task_model: ModelConfig
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    pool: PoolConfig
    run: RunConfig = RunConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrefPOConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
