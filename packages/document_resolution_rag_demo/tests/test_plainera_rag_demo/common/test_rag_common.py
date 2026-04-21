import pytest

from rag_demo.common import FixedWindowChunker, DemoDocument


def _doc(
    text: str,
    *,
    document_id: str = "doc-1",
    name: str = "Test document",
) -> DemoDocument:
    return DemoDocument(
        document_id=document_id,
        name=name,
        text=text,
    )


class TestFixedWindowChunkerInit:
    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            FixedWindowChunker(chunk_size=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            FixedWindowChunker(chunk_size=10, chunk_overlap=-1)

    def test_rejects_overlap_equal_to_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            FixedWindowChunker(chunk_size=10, chunk_overlap=10)

    def test_rejects_overlap_greater_than_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            FixedWindowChunker(chunk_size=10, chunk_overlap=11)


class TestFixedWindowChunkerChunkDocuments:
    def test_returns_empty_for_empty_input(self) -> None:
        chunker = FixedWindowChunker(chunk_size=5)

        out = chunker.chunk_documents([])

        assert out == []

    def test_returns_empty_for_empty_document_text(self) -> None:
        chunker = FixedWindowChunker(chunk_size=5)

        out = chunker.chunk_documents([_doc("")])

        assert out == []

    def test_skips_whitespace_only_document(self) -> None:
        chunker = FixedWindowChunker(chunk_size=5)

        out = chunker.chunk_documents([_doc("     \n   ")])

        assert out == []

    def test_emits_single_chunk_when_text_fits_in_one_window(self) -> None:
        chunker = FixedWindowChunker(chunk_size=10)

        out = chunker.chunk_documents([_doc("hello")])

        assert len(out) == 1
        chunk = out[0]
        assert chunk.chunk_id == "doc-1:0"
        assert chunk.document_id == "doc-1"
        assert chunk.document_name == "Test document"
        assert chunk.ordinal == 0
        assert chunk.start_offset == 0
        assert chunk.end_offset == 5
        assert chunk.text == "hello"

    def test_emits_fixed_windows_without_overlap(self) -> None:
        chunker = FixedWindowChunker(chunk_size=4, chunk_overlap=0)

        out = chunker.chunk_documents([_doc("abcdefghij")])

        assert [(c.ordinal, c.start_offset, c.end_offset, c.text) for c in out] == [
            (0, 0, 4, "abcd"),
            (1, 4, 8, "efgh"),
            (2, 8, 10, "ij"),
        ]
        assert [c.chunk_id for c in out] == ["doc-1:0", "doc-1:1", "doc-1:2"]

    def test_emits_fixed_windows_with_overlap(self) -> None:
        chunker = FixedWindowChunker(chunk_size=4, chunk_overlap=1)

        out = chunker.chunk_documents([_doc("abcdefghij")])

        assert [(c.ordinal, c.start_offset, c.end_offset, c.text) for c in out] == [
            (0, 0, 4, "abcd"),
            (1, 3, 7, "defg"),
            (2, 6, 10, "ghij"),
        ]

    def test_preserves_input_document_order(self) -> None:
        chunker = FixedWindowChunker(chunk_size=3)

        out = chunker.chunk_documents(
            [
                _doc("abcdef", document_id="doc-a", name="A"),
                _doc("wxyz", document_id="doc-b", name="B"),
            ]
        )

        assert [(c.document_id, c.chunk_id, c.text) for c in out] == [
            ("doc-a", "doc-a:0", "abc"),
            ("doc-a", "doc-a:1", "def"),
            ("doc-b", "doc-b:0", "wxy"),
            ("doc-b", "doc-b:1", "z"),
        ]

    def test_skips_whitespace_only_windows_but_keeps_ordinals_for_emitted_chunks(self) -> None:
        chunker = FixedWindowChunker(chunk_size=3, chunk_overlap=0)

        out = chunker.chunk_documents([_doc("abc   def")])

        assert [(c.ordinal, c.start_offset, c.end_offset, c.text) for c in out] == [
            (0, 0, 3, "abc"),
            (2, 6, 9, "def"),
        ]
        assert [c.chunk_id for c in out] == ["doc-1:0", "doc-1:2"]

    def test_last_chunk_stops_at_document_end(self) -> None:
        chunker = FixedWindowChunker(chunk_size=6, chunk_overlap=2)

        out = chunker.chunk_documents([_doc("abcdefghijk")])

        assert [(c.start_offset, c.end_offset, c.text) for c in out] == [
            (0, 6, "abcdef"),
            (4, 10, "efghij"),
            (8, 11, "ijk"),
        ]
