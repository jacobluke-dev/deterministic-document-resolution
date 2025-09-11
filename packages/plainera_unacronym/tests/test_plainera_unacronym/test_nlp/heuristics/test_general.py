# tests/unit/test_at_sentence_boundary.py
import pytest

from plainera_unacronym.nlp.heuristics.general import at_sentence_boundary


def _extract(text_with_caret: str) -> tuple[str, int]:
    """Turn '...^...' into (text, pos)."""
    pos = text_with_caret.index("^")
    return text_with_caret.replace("^", ""), pos


@pytest.mark.unit
class TestAtSentenceBoundary:
    @pytest.mark.parametrize(
        "sample, expected",
        [
            ("^Hello", True),                                   # start of doc
            ("Hello. ^World", True),                            # period + space
            ("Hello.^World", True),                             # period, no space
            ('He said: "Go." ^Then', True),                     # period + closing quote
            ("Do it now.) ^Then", True),                        # period + ) closer
            ('"Go?!" ^Next', True),                             # mixed ?!
            ("Wait… ^Go", True),                                # unicode ellipsis
            ("Hello ^brave world", False),                      # mid-sentence
            ("Wait—^no", False),                                # em dash is not a terminator
            ("Hello\n^World", False),                           # newline alone isn't a boundary
            ('He said "hello" ^and left.', False),              # closer without terminator
            ('He said: “Go.”^Then', True),                      # curly closer right before next
            ('Done! ”^Next', True),                             # space + curly quote closer
            ('Done! »^Next', True),                             # guillemet closer
            ("U.S.^Policy", True),  # dotted initialism, no space
            ("Hello\t\t^World", False),  # tabs as whitespace
            ("Dog.     ^The", True)
        ],
    )
    def test_various(self, sample: str, expected: bool):
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos) is expected

    def test_whitespace_then_closers_then_terms(self):
        # e.g. space(s) -> closers -> terminator cluster -> next token
        sample = 'He: "Done?!"   ^Next'
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos) is True

    def test_many_spaces_after_terminator(self):
        sample = "End.     ^Next"
        text, pos = _extract(sample)
        assert at_sentence_boundary(text, pos) is True
