#!/bin/bash
# ====================================================================
#  One-click runner for macOS -- double-click this file.
#  (First time: you may need to allow it under System Settings >
#   Privacy & Security, or run `chmod +x run_all.command` once.)
# ====================================================================
cd "$(dirname "$0")"
python3 run_all.py "$@"
echo
read -n 1 -s -r -p "Done. Press any key to close."
