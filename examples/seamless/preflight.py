"""Preflight checks for the seamless example suite."""

import os
from importlib.util import find_spec


def _is_set(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip())


if __name__ == "__main__":
    has_openai = _is_set("OPENAI_API_KEY")
    has_gemini = _is_set("GEMINI_API_KEY") or _is_set("GOOGLE_API_KEY")
    has_ifeval = find_spec("instruction_following_eval") is not None

    print("=== PrefPO seamless preflight ===")
    print(f"OPENAI_API_KEY: {'OK' if has_openai else 'MISSING'}")
    print(f"GEMINI_API_KEY/GOOGLE_API_KEY: {'OK' if has_gemini else 'MISSING'}")
    print(f"instruction_following_eval package: {'OK' if has_ifeval else 'MISSING'}")

    missing = []
    if not has_openai:
        missing.append("OPENAI_API_KEY")
    if not has_gemini:
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY")
    if not has_ifeval:
        missing.append("instruction_following_eval")

    if missing:
        print("\nMissing requirements:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nAll seamless suite prerequisites are available.")
