"""Core data types for the PrefPO optimization pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

# JSON-serializable value type for metadata fields.
JsonValue = Union[str, int, float, bool, None]

_ALLOWED_TYPES = (str, int, float, bool, type(None))


def _validate_metadata(metadata: dict[str, JsonValue]) -> None:
    """Raise TypeError if metadata contains non-JSON-serializable values."""
    for key, val in metadata.items():
        if not isinstance(key, str):
            raise TypeError(f"Metadata key must be str, got {type(key).__name__}")
        if not isinstance(val, _ALLOWED_TYPES):
            raise TypeError(
                f"Metadata value for '{key}' must be str|int|float|bool|None, "
                f"got {type(val).__name__}"
            )


class PromptRole(str, Enum):
    USER = "user"
    SYSTEM = "system"


@dataclass
class Prompt:
    """A prompt being optimized."""

    value: str
    role: PromptRole
    name: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)


@dataclass(frozen=True)
class Sample:
    """A single evaluation sample (question + expected answer)."""

    index: int
    question: str
    target: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)


@dataclass(frozen=True)
class ModelOutput:
    """Output from running a prompt through the task model."""

    sample_index: int
    prompt_sent: str
    response: str


@dataclass(frozen=True)
class DiscriminatorResult:
    """Result from the discriminator comparing two prompt variants."""

    preferred: int  # 1 or 2
    feedback: str


@dataclass(frozen=True)
class IterationRecord:
    """Record of a single optimization iteration for logging."""

    iteration: int
    prompt_a: Prompt
    prompt_b: Prompt
    prompt_a_score: float | None
    prompt_b_score: float | None
    preferred: int
    feedback: str
    improved_prompt: Prompt
    improved_score: float
    discriminator_usage: dict[str, int]
    optimizer_usage: dict[str, int]
