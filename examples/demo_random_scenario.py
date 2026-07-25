"""Thin wrapper around :mod:`sigpa.demo` kept for repository browsing.

Usage:  python examples/demo_random_scenario.py [nodes] [pois] [seed]
(equivalent to the ``sigpa-demo`` console command after installation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigpa.demo import main

if __name__ == "__main__":
    main()
