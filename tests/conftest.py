from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_MAP = ROOT / "maps" / "test"


@pytest.fixture
def test_map() -> Path:
    return TEST_MAP
