import pytest
import os
import sys
from pathlib import Path

CWD = Path(os.path.dirname(os.path.realpath(__file__)))
SRC = CWD.parent / "maet_pln"
sys.path.append(str(SRC))

# if using mac need this
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
