"""Base grader interface and GradeResult data type."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from prefpo.config import ModelConfig
from prefpo.types import Prompt, Sample


@dataclass(frozen=True)
class GradeResult:
    """Result from grading a prompt against samples."""

    score: float
    n: int
    stderr: float = 0.0
    per_sample: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)


class Grader(ABC):
    """Abstract base class for graders.

    Graders own the full evaluation pipeline: generate responses + check them.
    Subclasses must implement grade(). Optionally override check_output() for
    standalone trajectory annotation.
    """

    @abstractmethod
    async def grade(
        self,
        prompt: Prompt,
        samples: list[Sample],
        model_config: ModelConfig,
        semaphore: asyncio.Semaphore,
    ) -> GradeResult:
        """Evaluate a prompt and return a score.

        Args:
            prompt: The prompt to evaluate.
            samples: Evaluation samples (empty list in standalone mode).
            model_config: Task model config.
            semaphore: Shared concurrency semaphore.
        """
        ...

    def check_output(
        self,
        output: str,
        prompt_text: str | None = None,
    ) -> dict | None:
        """Lightweight per-output annotation for standalone trajectory building.

        Called by build_standalone_trajectory() when show_target=True.
        prompt_text is the prompt used to produce the output.
        Returns a dict of annotation fields or None if no annotation available.
        Default: returns None. Subclasses override to provide grading logic.
        """
        return None
