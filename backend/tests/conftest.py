import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
FIXTURE = ROOT / "backend/tests/fixtures/template.json"


@pytest.fixture(scope="session", autouse=True)
def opencv_threads():
    cv2.setNumThreads(2)
