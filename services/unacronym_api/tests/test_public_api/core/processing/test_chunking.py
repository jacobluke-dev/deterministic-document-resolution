from public_api.core.processing.acronym_chunking import make_chunks, merge_blocks, shift_blocks


def test_make_chunks_basic_overlap():
    text = "x" * 100
    chunks = make_chunks(text, chunk_size=30, overlap=10)

    assert chunks[0].start == 0 and chunks[0].end == 30
    assert chunks[1].start == 20 and chunks[1].end == 50
    assert chunks[-1].end == 100
    # coverage: no gaps
    assert chunks[0].start == 0
    assert all(chunks[i].end >= chunks[i + 1].start for i in range(len(chunks) - 1))


def test_shift_blocks_occurrences_and_definitions():
    blocks = [
        {
            "acronym": "MP",
            "occurrences": [{"start": 5, "end": 7}],
            "definitions": [{"text": "Member of Parliament", "start": 20, "end": 40}],
        }
    ]
    out = shift_blocks(blocks, 100)
    assert out[0]["occurrences"][0]["start"] == 105
    assert out[0]["definitions"][0]["start"] == 120


def test_merge_blocks_dedupes_and_orders():
    b1 = [{"acronym": "MP", "occurrences": [{"start": 10, "end": 12}], "definitions": []}]
    b2 = [{"acronym": "MP", "occurrences": [{"start": 10, "end": 12}], "definitions": []}]  # dup
    b3 = [{"acronym": "AI", "occurrences": [{"start": 5, "end": 7}], "definitions": []}]

    out = merge_blocks([b1, b2, b3])

    # order by first occurrence start then acronym
    assert out[0]["acronym"] == "AI"
    assert out[1]["acronym"] == "MP"
    # dedup occurrences
    assert len(out[1]["occurrences"]) == 1
