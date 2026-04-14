"""Crimson Desert Model Browser -- entry point.

Launch: python pac_browser.py

Modules live in browser/ package. This file provides:
- Entry point (if __name__ == '__main__')
- Re-exports for backward compatibility
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
UNPACKER_DIR = os.path.join(ROOT_DIR, "lazorr410-unpacker", "python")
sys.path.insert(0, UNPACKER_DIR)
sys.path.insert(0, SRC_DIR)

# Re-exports for backward compatibility (tests import these from pac_browser)
from browser.loaders import load_pac_mesh, read_pac_bytes, load_pam_mesh, read_pam_bytes


def main():
    from browser.main_window import main as _main
    _main()


if __name__ == "__main__":
    main()
