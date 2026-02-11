"""Result file writer for PrefPO optimization runs."""

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from prefpo.config import PrefPOConfig
from prefpo.types import IterationRecord


class RunLogger:
    """Writes config, iteration history, final pool, and summary to disk."""

    def __init__(
        self,
        config: PrefPOConfig,
        base_dir: Path | None = None,
        trial_index: int | None = None,
    ) -> None:
        if base_dir is None:
            base_dir = Path(config.run.output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        if trial_index is not None:
            self._run_id = f"run_{timestamp}_{trial_index}_{suffix}"
        else:
            self._run_id = f"run_{timestamp}_{suffix}"
        self._run_dir = base_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        config_path = self._run_dir / "config.json"
        config_path.write_text(json.dumps(config.model_dump(), indent=2))

        # Prepare iteration history file
        self._history_path = self._run_dir / "iteration_history.jsonl"

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_iteration(self, record: IterationRecord) -> None:
        """Append one iteration record as a JSON line."""
        data = asdict(record)
        with open(self._history_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_final_pool(self, pool_state: dict) -> None:
        """Write the final pool state to disk."""
        path = self._run_dir / "final_pool_state.json"
        path.write_text(json.dumps(pool_state, indent=2))

    def log_grading_outputs(
        self,
        iteration: int,
        prompt_name: str,
        role: str,
        score: float,
        outputs: list[dict],
        per_sample: list[dict],
    ) -> None:
        """Append grading outputs as a JSON line to grading_outputs.jsonl.

        Merges raw model outputs with per-sample grading metadata by index.
        """
        # Merge outputs and per_sample by position
        merged = []
        for i, out in enumerate(outputs):
            entry = dict(out)
            if i < len(per_sample):
                entry.update(per_sample[i])
            merged.append(entry)

        record = {
            "iteration": iteration,
            "prompt_name": prompt_name,
            "role": role,
            "score": score,
            "outputs": merged,
        }
        path = self._run_dir / "grading_outputs.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_summary(self, summary: dict) -> None:
        """Write the run summary to disk."""
        path = self._run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2))
