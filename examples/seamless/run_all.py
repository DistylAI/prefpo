"""Run all seamless examples sequentially."""

import subprocess
import sys
import time
from pathlib import Path


SCRIPTS = [
    "preflight.py",
    "minimal_instruction_smoke.py",
    "minimal_standalone_smoke.py",
    "from_yaml_bbh.py",
    "cli_bbh.py",
    "optimize_async_direct.py",
    "optimize_multi_trial_direct.py",
    "bbh_binary_exact.py",
    "gemini_provider.py",
    "instruction_llm_judge.py",
]


if __name__ == "__main__":
    suite_dir = Path(__file__).resolve().parent
    root = suite_dir.parents[1]
    results = []

    print("=== Running seamless suite ===")
    for idx, script_name in enumerate(SCRIPTS, start=1):
        script_path = suite_dir / script_name
        print(f"\n[{idx}/{len(SCRIPTS)}] {script_name}")
        started = time.time()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=root,
            check=False,
        )
        elapsed = time.time() - started
        passed = completed.returncode == 0
        results.append((script_name, passed, elapsed, completed.returncode))
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")

    print("\n=== Seamless suite summary ===")
    failed = 0
    for script_name, passed, elapsed, code in results:
        if passed:
            print(f"PASS {script_name} ({elapsed:.1f}s)")
        else:
            failed += 1
            print(f"FAIL {script_name} ({elapsed:.1f}s, exit={code})")

    if failed:
        print(f"\nFinished with {failed} failing script(s).")
        raise SystemExit(1)

    print("\nAll seamless scripts passed.")
