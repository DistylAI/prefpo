"""Run the PrefPO CLI with a tiny BBH config."""

import subprocess
import sys
import tempfile
from pathlib import Path


CONFIG_TEXT = """mode: "instruction"
task_model:
  name: "openai/gpt-4o-mini"
discriminator:
  model:
    name: "openai/gpt-4.1"
    is_reasoning: false
  show_expected: true
  criteria: "correctness of the final answer"
optimizer:
  model:
    name: "openai/gpt-4.1"
    is_reasoning: false
  constraints: "Do not remove the ANSWER format."
pool:
  initial_prompts:
    - "Answer the multiple choice question and end with 'ANSWER: $LETTER'."
    - "Think briefly, then answer and end with 'ANSWER: $LETTER'."
  prompt_role: "user"
  update_strategy: "replace"
run:
  iterations: 1
  max_concurrent: 1
  output_dir: "results/seamless/cli_bbh"
"""


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "cli_bbh.yaml"
        config_path.write_text(CONFIG_TEXT, encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "prefpo",
            "--config",
            str(config_path),
            "--dataset",
            "bbh",
            "--subset",
            "disambiguation_qa",
            "--train-size",
            "2",
            "--val-size",
            "2",
            "--test-size",
            "2",
            "--seed",
            "42",
        ]

        completed = subprocess.run(cmd, cwd=root, check=False)
        raise SystemExit(completed.returncode)
