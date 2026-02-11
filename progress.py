"""Rich-based progress display for PrefPO optimization runs."""

from __future__ import annotations

import sys
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def _is_interactive() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


# ---------------------------------------------------------------------------
# Single-trial display
# ---------------------------------------------------------------------------


class ProgressDisplay:
    """Progress bar + status line for a single optimization run."""

    def __init__(self, iterations: int, verbose: bool = True):
        self._rich = verbose and _is_interactive()
        self._text = verbose and not _is_interactive()
        self._iterations = iterations
        self._progress: Progress | None = None
        self._task_id = None
        self._console = Console(stderr=True) if self._rich else None

    def start(self) -> None:
        if not self._rich:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Optimizing"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("best: {task.fields[best]:.3f}"),
            console=self._console,
        )
        self._task_id = self._progress.add_task(
            "optimize", total=self._iterations, best=0.0,
        )
        self._progress.start()

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def set_status(self, text: str) -> None:
        if self._progress is None:
            return
        self._progress.update(self._task_id, description=f"[dim]  ↳ {text}")

    def complete_iteration(
        self, iteration: int, score: float, best: float, preferred: int,
    ) -> None:
        if self._rich and self._progress is not None:
            self._progress.update(self._task_id, advance=1, best=best)
            arrow = "↑" if score >= best else " "
            self._progress.console.print(
                f"  [dim]Iter {iteration + 1}  preferred={preferred}  "
                f"improved={score:.3f} {arrow}[/dim]"
            )
        elif self._text:
            arrow = "^" if score >= best else " "
            print(
                f"  Iter {iteration + 1}/{self._iterations}  "
                f"preferred={preferred}  improved={score:.3f}  "
                f"best={best:.3f} {arrow}"
            )

    def finish(
        self,
        best_score: float,
        test_score: float | None,
        prompt_name: str,
        run_id: str,
        results_dir: str | None = None,
    ) -> None:
        self.stop()
        test_str = f"  (test: {test_score:.3f})" if test_score is not None else ""
        if self._rich and self._console is not None:
            self._console.print(f"\n[bold green]✓ Optimization complete[/bold green]")
            self._console.print(f"  Best score: {best_score:.3f}{test_str}")
            self._console.print(f"  Best prompt: {prompt_name}")
            self._console.print(f"  Run ID: {run_id}")
            if results_dir:
                self._console.print(f"  Results saved to: {results_dir}")
        elif self._text:
            print(f"\nOptimization complete")
            print(f"  Best score: {best_score:.3f}{test_str}")
            print(f"  Best prompt: {prompt_name}")
            print(f"  Run ID: {run_id}")
            if results_dir:
                print(f"  Results saved to: {results_dir}")


# ---------------------------------------------------------------------------
# Multi-trial display
# ---------------------------------------------------------------------------


class MultiTrialDisplay:
    """One progress bar per trial, all visible simultaneously."""

    def __init__(self, n_trials: int, iterations: int, verbose: bool = True):
        self._rich = verbose and _is_interactive()
        self._text = verbose and not _is_interactive()
        self._n_trials = n_trials
        self._iterations = iterations
        self._progress: Progress | None = None
        self._task_ids: list = []
        self._console = Console(stderr=True) if self._rich else None

    def start(self) -> None:
        if not self._rich:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("best: {task.fields[best]:.3f}"),
            TextColumn("{task.fields[status]}"),
            console=self._console,
        )
        for i in range(self._n_trials):
            tid = self._progress.add_task(
                f"trial_{i}",
                total=self._iterations,
                label=f"[bold]Trial {i + 1}/{self._n_trials}[/bold]",
                best=0.0,
                status="",
            )
            self._task_ids.append(tid)
        self._progress.start()

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def update_trial(self, trial_idx: int, iteration: int, best_score: float) -> None:
        """Called after each iteration within a trial completes."""
        if self._rich and self._progress is not None and trial_idx < len(self._task_ids):
            tid = self._task_ids[trial_idx]
            self._progress.update(tid, completed=iteration + 1, best=best_score)
        elif self._text:
            print(
                f"  Trial {trial_idx + 1}/{self._n_trials}  "
                f"iter {iteration + 1}/{self._iterations}  "
                f"best={best_score:.3f}"
            )

    def complete_trial(self, trial_idx: int, best_score: float) -> None:
        if self._rich and self._progress is not None and trial_idx < len(self._task_ids):
            tid = self._task_ids[trial_idx]
            self._progress.update(
                tid, completed=self._iterations, best=best_score,
                status="[bold green]✓[/bold green]",
            )
        elif self._text:
            print(
                f"  Trial {trial_idx + 1}/{self._n_trials} complete  "
                f"best={best_score:.3f}"
            )

    def make_callback(self, trial_idx: int) -> Callable[[int, float], None]:
        """Return a callback for optimize_async to call after each iteration."""
        def cb(iteration: int, best_score: float) -> None:
            self.update_trial(trial_idx, iteration, best_score)
        return cb

    def finish(
        self,
        mean_val: float,
        std_val: float,
        best_score: float,
        mean_test: float | None = None,
        std_test: float | None = None,
        results_dir: str | None = None,
    ) -> None:
        self.stop()
        test_str = ""
        if mean_test is not None and std_test is not None:
            test_str = f"  test: {mean_test:.3f} +/- {std_test:.3f}"
        if self._rich and self._console is not None:
            self._console.print(f"\n[bold green]✓ Multi-trial complete[/bold green]")
            self._console.print(f"  Val: {mean_val:.3f} ± {std_val:.3f}{test_str}")
            self._console.print(f"  Best trial score: {best_score:.3f}")
            if results_dir:
                self._console.print(f"  Results saved to: {results_dir}")
        elif self._text:
            print(f"\nMulti-trial complete")
            print(f"  Val: {mean_val:.3f} +/- {std_val:.3f}{test_str}")
            print(f"  Best trial score: {best_score:.3f}")
            if results_dir:
                print(f"  Results saved to: {results_dir}")
