"""Entry point for python -m prefpo."""

import asyncio

from prefpo.cli import main

if __name__ == "__main__":
    asyncio.run(main())
