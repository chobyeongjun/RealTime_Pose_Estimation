"""pytest 설정 — src/ path 자동 추가.

`pytest tests/` 또는 `python3 -m pytest tests/` 모두 OK.
"""
import os
import sys

# tests/ 의 sibling = src/perception/...
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
