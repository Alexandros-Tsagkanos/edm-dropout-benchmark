#!/bin/bash
# ====================================================================
#  One-click runner for Linux -- run `./run_all.sh`
#  (First time: `chmod +x run_all.sh`. If you get a bad-interpreter
#   error from CRLF line endings, just run `python3 run_all.py`.)
# ====================================================================
cd "$(dirname "$0")"
python3 run_all.py "$@"
