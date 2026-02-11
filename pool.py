"""Prompt pool management with scoring and sampling."""

import random
from typing import Any

from prefpo.types import Prompt


class PromptPool:
    """Manages a pool of candidate prompts with scores and pair sampling."""

    def __init__(self, prompts: list[Prompt], seed: int = 42) -> None:
        if not prompts:
            raise ValueError("PromptPool requires at least one prompt")

        self.entries: list[Prompt] = []
        self.scores: dict[tuple[str, str], float] = {}  # (value, role) -> score
        self._rng = random.Random(seed)

        for i, p in enumerate(prompts):
            name = p.name or f"seed_{i}"
            self.entries.append(
                Prompt(value=p.value, role=p.role, name=name, metadata=p.metadata)
            )

    def sample_pair(self) -> tuple[Prompt, Prompt]:
        """Sample two distinct prompts from the pool."""
        if len(self.entries) < 2:
            raise ValueError("Need at least 2 prompts to sample a pair")
        pair = self._rng.sample(self.entries, 2)
        return pair[0], pair[1]

    def add(self, prompt: Prompt) -> None:
        """Add a new prompt to the pool."""
        self.entries.append(prompt)

    def replace_non_preferred(
        self, new: Prompt, a_name: str, b_name: str, preferred: int
    ) -> None:
        """Replace the non-preferred prompt with the new one."""
        non_pref_name = b_name if preferred == 1 else a_name
        for i, entry in enumerate(self.entries):
            if entry.name == non_pref_name:
                self.entries[i] = new
                return
        raise ValueError(
            f"Cannot replace: prompt '{non_pref_name}' not found in pool. "
            f"Pool names: {[e.name for e in self.entries]}"
        )

    def set_score(self, prompt: Prompt, score: float) -> None:
        """Cache a score keyed by (text, role) to avoid redundant evals."""
        self.scores[(prompt.value, prompt.role.value)] = score

    def get_score(self, prompt: Prompt) -> float | None:
        """Get cached score for a prompt, or None if not scored."""
        return self.scores.get((prompt.value, prompt.role.value))

    def best(self) -> Prompt:
        """Return the prompt with the highest score."""
        scored = [(p, self.scores.get((p.value, p.role.value))) for p in self.entries]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            return self.entries[0]
        return max(scored, key=lambda x: x[1])[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize pool state for logging."""
        return {
            "prompts": [
                {
                    "name": p.name,
                    "value": p.value,
                    "role": p.role.value,
                    "score": self.scores.get((p.value, p.role.value)),
                }
                for p in self.entries
            ]
        }
