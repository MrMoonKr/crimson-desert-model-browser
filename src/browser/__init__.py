"""Crimson Desert Model Browser package."""

import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_UNPACKER_DIR = os.path.join(_ROOT_DIR, "lazorr410-unpacker", "python")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _UNPACKER_DIR not in sys.path:
    sys.path.insert(0, _UNPACKER_DIR)
