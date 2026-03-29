from public_api.core.processing.acronym_chunking import merge_acronym_blocks, shift_acronym_blocks


def test_shift_blocks_occurrences_and_definitions():
    blocks = [
        {
            "acronym": "MP",
            "occurrences": [{"start": 5, "end": 7}],
            "definitions": [{"text": "Member of Parliament", "start": 20, "end": 40}],
        }
    ]
    out = shift_acronym_blocks(blocks, 100)
    assert out[0]["occurrences"][0]["start"] == 105
    assert out[0]["definitions"][0]["start"] == 120


def test_merge_blocks_dedupes_and_orders():
    b1 = [{"acronym": "MP", "occurrences": [{"start": 10, "end": 12}], "definitions": []}]
    b2 = [{"acronym": "MP", "occurrences": [{"start": 10, "end": 12}], "definitions": []}]  # dup
    b3 = [{"acronym": "AI", "occurrences": [{"start": 5, "end": 7}], "definitions": []}]

    out = merge_acronym_blocks([b1, b2, b3])

    # order by first occurrence start then acronym
    assert out[0]["acronym"] == "AI"
    assert out[1]["acronym"] == "MP"
    # dedup occurrences
    assert len(out[1]["occurrences"]) == 1
