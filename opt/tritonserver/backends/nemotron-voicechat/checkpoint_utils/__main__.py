"""
Entry point for `python -m checkpoint_utils` and `python -m riva.cbe.s2s.checkpoint_utils`.
"""

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from .cli import main

if __name__ == "__main__":
    main()
