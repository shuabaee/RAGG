from core.pdf_utils import paper_id_for


def test_paper_id_sees_changes_after_first_200kb():
    head = b"%PDF-1.7 " + b"x" * 300_000
    assert paper_id_for("thesis.pdf", head + b"version 1") != paper_id_for("thesis.pdf", head + b"version 2")


def test_paper_id_is_stable():
    assert paper_id_for("a.pdf", b"same bytes") == paper_id_for("a.pdf", b"same bytes")
