#!/bin/zsh
# Double-click this file in Finder to launch DumbConvert.
# MallocSpaceEfficient: macOS gives freed picture memory back at once (see main() in simpleneg/app.py)
cd "$(dirname "$0")" && MallocSpaceEfficient=1 .venv/bin/python -m simpleneg
