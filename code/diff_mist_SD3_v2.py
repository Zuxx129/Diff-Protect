#!/usr/bin/env python3
"""Stable SD3 v2 command entry.

For now this module delegates to the existing SD3 launcher. The experiment
contract is enforced by configs and scripts while the legacy entry remains
available for compatibility.
"""
from diff_mist_SD3 import main


if __name__ == "__main__":
    main()
