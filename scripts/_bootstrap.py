"""Importable bootstrap for repo scripts -- ``import _bootstrap`` as the first line.

Python puts a script's own directory on ``sys.path[0]`` when you run
``python scripts/foo.py``, so this module is importable from any script in
``scripts/`` with no path hack. It puts the repo ROOT on ``sys.path`` (so
``import src...`` resolves) and makes stdout UTF-8 (Windows consoles default to
cp1252 and choke on the unicode in our prints).

It deliberately does NOT touch matplotlib. The headless ("Agg") backend is set
only inside ``scripts/run.py`` via ``src.repro.headless``, so importing any
library or experiment module from a notebook never flips the backend -- killing
the old "import a script in a notebook and %matplotlib inline silently dies"
trap at the root.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
