import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import vectorstore  # noqa: E402
from core.models import Chunk  # noqa: E402
from core.retrieval import tokenize  # noqa: E402


def make_chunks(paper_id: str, texts: list[str], page: int = 1, section: str = "results") -> list[Chunk]:
    return [
        Chunk(f"{paper_id}-{i}", paper_id, f"{paper_id}.pdf", page, section, t, tokenize(t))
        for i, t in enumerate(texts, 1)
    ]


@pytest.fixture
def store(tmp_path):
    vectorstore.reset_client_for_tests(str(tmp_path / "chroma"))
    yield vectorstore
    vectorstore.reset_client_for_tests(None)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: downloads/runs the real FlashRank model")
