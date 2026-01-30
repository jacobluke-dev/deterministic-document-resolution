import pytest

from plainera_unacronym.nlp.extraction.matchers.common import split_compound, initials_seq, is_mixed_case_acronym, \
    match_from


class TestMatchFrom:
    def test_exact_match_from_zero(self):
        letters = list("PDFX")
        acronym = list("PDF")
        end, used = match_from(letters, acronym, 0)
        assert end == 3
        assert used == [0, 1, 2]

    def test_no_match(self):
        letters = list("PFX")
        acronym = list("PDF")
        assert match_from(letters, acronym, 0) is None

    def test_start_offset(self):
        letters = list("APDFZ")
        acronym = list("PDF")
        # Starting at 1 should match P(1), D(2), F(3) → end index 4
        end, used = match_from(letters, acronym, 1)
        assert end == 4
        assert used == [1, 2, 3]

    def test_start_offset_scans_forward_not_strict_anchor(self):
        letters = list("ABCABC")
        acronym = list("ABC")
        # From start=0 it matches at [0,1,2]
        assert match_from(letters, acronym, 0) == (3, [0, 1, 2])
        # From start=1 it can skip to the next 'A' and match at [3,4,5]
        assert match_from(letters, acronym, 1) == (6, [3, 4, 5])

    def test_greedy_end_index_is_exclusive(self):
        letters = list("AXBYCZD")
        acronym = list("ABCD")
        # Match A(0), B(2), C(4), D(6) → last match at 6; end should be 7
        end, used = match_from(letters, acronym, 0)
        assert used == [0, 2, 4, 6]
        assert end == 7  # exclusive

    def test_letters_shorter_than_acronym(self):
        letters = list("PD")
        acronym = list("PDF")
        assert match_from(letters, acronym, 0) is None

    def test_empty_acronym_matches_immediately(self):
        letters = list("ANY")
        acronym = []  # empty target
        end, used = match_from(letters, acronym, 2)
        # With empty acronym, loop doesn't run; returns (start, [])
        assert (end, used) == (2, [])

    def test_start_past_end_returns_none(self):
        letters = list("PDF")
        acronym = list("P")
        assert match_from(letters, acronym, 5) is None

    def test_case_sensitive(self):
        letters = list("Pdf")
        acronym = list("PDF")
        # Exact equality is required; pipeline should uppercase upstream
        assert match_from(letters, acronym, 0) is None


class TestSplitCompound:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("GPU", ["GPU"]),  # no split
            ("read-only", ["read", "only"]),  # hyphen
            ("C/CPP", ["C", "CPP"]),  # slash
            ("U.S.A.", ["U", "S", "A", ""]),  # NOTE: trailing '' would be filtered out by impl
            ("Foo.Bar", ["Foo", "Bar"]),  # dot
            ("R&D", ["R", "D"]),  # ampersand
            ("A-B/C.D&E", ["A", "B", "C", "D", "E"]),  # mixed delimiters
            ("--GPU--", ["", "GPU", ""]),  # leading/trailing delimiters (empties dropped)
            ("a--b", ["a", "", "b"]),  # repeated delimiter (middle '' dropped)
            ("", []),  # empty token -> []
            ("----", []),  # only delimiters -> []
            ("co-op", ["co", "op"]),  # splits on '-'
            ("Queen’s", ["Queen’s"]),  # apostrophe does not split
            ("snake_case", ["snake_case"]),  # underscore does not split
            ("β-blocker", ["β", "blocker"]),  # Unicode letters + hyphen
            ("3D-Print", ["3D", "Print"]),  # alnum pieces
            ("A&B&C", ["A", "B", "C"]),  # multiple &
            ("v1.2.3", ["v1", "2", "3"]),  # dot with numbers
            ("Hypertext", ["Hyper", "text"]),
            ("HyperText", ["Hyper", "text"])
        ],
    )
    def test_split_various(self, token, expected):
        # Filtered empties: replicate function’s behavior for cases where parametrization shows '' parts
        out = split_compound(token)
        assert out == [p for p in expected if p]

    def test_repeated_mixed_delimiters(self):
        token = "a--b///c..d&&e"
        assert split_compound(token) == ["a", "b", "c", "d", "e"]



    @pytest.mark.parametrize(
        "token, expected",
        [
            # ---- Policy A: keep only digit-leading letters OR single-letter + trailing digits ----
            ("3D", ["3D"]),
            ("2FA", ["2FA"]),
            ("7Zip", ["7Zip"]),
            ("v1", ["v1"]),
            ("x86", ["x86"]),

            # ---- Policy A: should split other letter+digit patterns ----
            ("HTTP2", ["HTTP2"]),
            ("RFC7231", ["RFC7231"]),
            ("SHA256", ["SHA256"]),
            ("H264", ["H264"]),
            ("B2B", ["B2B"]),

            # ---- separators: hyphen/slash/dot/& ----
            ("Foo-Bar", ["Foo", "Bar"]),
            ("Foo/Bar", ["Foo", "Bar"]),
            ("Foo.Bar", ["Foo", "Bar"]),
            ("Foo&Bar", ["Foo", "Bar"]),

            # ---- CamelCase splitting (ASCII) ----
            ("XMLHttpRequest", ["XML", "Http", "Request"]),
            ("MyThing", ["My", "Thing"]),
            ("thing", ["thing"]),
        ],
    )
    def test_split_compound_policy_a(self, token, expected):
        assert split_compound(token) == expected


    def test_split_compound_non_ascii_kept_intact(self):
        # Non-ASCII => do not Camel-split; keep the piece intact
        assert split_compound("ÅngströmValue") == ["ÅngströmValue"]


    @pytest.mark.parametrize(
        "token, expected_prefix",
        [
            # these are in LEXICAL_SPLITS in your snippet
            ("websocket", ["Web", "Socket"]),
            ("middleware", ["Middle", "Ware"]),
            ("firmware", ["Firm", "Ware"]),
            ("hardware", ["Hard", "Ware"]),
            ("software", ["Soft", "Ware"]),
            ("hostname", ["Host", "Name"]),
            ("password", ["Pass", "Word"]),
            ("database", ["Data", "Base"]),
            ("typescript", ["Type", "Script"]),
            ("powershell", ["Power", "Shell"]),
            ("bitbucket", ["Bit", "Bucket"]),
            ("gitlab", ["Git", "Lab"]),
            ("github", ["Git", "Hub"]),
            ("postgresql", ["Postgres", "SQL"]),
            ("mysql", ["My", "SQL"]),
            ("mssql", ["MS", "SQL"]),
            ("newline", ["New", "Line"]),
            ("filepath", ["File", "Path"]),
            ("filename", ["File", "Name"]),
            ("checksum", ["Check", "Sum"]),
        ],
    )
    def test_split_compound_lexical_splits(self, token, expected_prefix):
        assert split_compound(token) == expected_prefix


class TestInitialsSeqUnit:
    def test_basic_three_tokens(self, _patch):
        # Force a single part per token; check letters+owners map 1:1 to tokens
        _patch(
            initials_seq,
            split_compound=lambda tok: [tok],
            re=__import__("re"),
        )
        tokens = ["Portable", "Document", "Format"]
        letters, owners = initials_seq(tokens)
        assert letters == ["P", "D", "F"]
        assert owners == [0, 1, 2]

    def test_compound_parts_emit_multiple_initials_with_same_owner(self, _patch):
        # Simulate "C++" -> ["C", "Plus", "Plus"] from the *same* token index
        parts_map = {
            "C++": ["C", "Plus", "Plus"],
            "GPU": ["GPU"],
        }
        _patch(
            initials_seq,
            split_compound=lambda tok: parts_map[tok],
            re=__import__("re"),
        )
        tokens = ["C++", "GPU"]
        letters, owners = initials_seq(tokens)
        assert letters == ["C", "P", "P", "G"]
        assert owners == [0, 0, 0, 1]

    def test_tokens_with_no_alnum_parts_are_ignored(self, _patch):
        # Parts with no [A-Za-z0-9] yield no initials
        _patch(
            initials_seq,
            split_compound=lambda tok: ["—", "…"],  # emdash, ellipsis
            re=__import__("re"),
        )
        tokens = ["—…"]
        letters, owners = initials_seq(tokens)
        assert letters == []
        assert owners == []

    def test_stopword_checked_before_split(self, _patch):
        # If the whole token is a stopword, we skip it entirely (no splitting)
        called = {"split": 0}

        def spy_split(tok):
            called["split"] += 1
            return [tok]  # would have produced something if not skipped

        _patch(initials_seq, split_compound=spy_split, re=__import__("re"))
        tokens = ["and-or", "Useful"]
        letters, owners = initials_seq(tokens)
        assert letters == ["A", "U"]
        assert owners == [0, 1]
        # Ensure split was *not* called for the stopword token
        assert called["split"] == 2  # only for "Useful"


class TestInitialsSeqIntegration:

    def test_plain_token_emits_single_initial_by_default(self):
        tokens = ["GPU"]
        letters, owners = initials_seq(tokens)
        assert letters == ["G"]
        assert owners == [0]

    def test_compound_splitting_and_digits(self):
        tokens = ["3/4-inch", "co-op", "R&D", "v1.2.3"]
        letters, owners = initials_seq(tokens)
        # Expected from real split_compound:
        # "3/4-inch" -> ["3","4","inch"]      -> 3,4,I (owners 0,0,0)
        # "co-op"    -> ["co","op"]           -> C,O   (owners 1,1)
        # "R&D"      -> ["R","D"]             -> R,D   (owners 2,2)
        # "v1.2.3"   -> ["v1","2","3"]        -> V,2,3 (owners 3,3,3)
        assert letters == ["3", "4", "I", "C", "O", "R", "D", "V", "2", "3"]
        assert owners == [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]

    def test_unicode_letters_in_tokens(self):
        tokens = ["β-blocker", "Ångström", "GPU"]
        letters, owners = initials_seq(tokens)
        # "β-blocker" -> parts ["β","blocker"] → first alpha is 'β' (Unicode) → 'Β' (Greek beta uppercase)
        # 2nd B is from blocker
        # "Ångström"  -> first alpha is 'Å'     → 'Å'
        # "GPU"       -> 'G'
        assert letters == ["Β", "B", "Å", "G"]  # Python uppercases β to Β
        assert owners == [0, 0, 1, 2]


    def test_initials_seq_expand_allcaps_supports_mixed_case_acronyms(self):
        # The core behavioural goal: mRNA should align against "messenger RNA"
        tokens = ["messenger", "RNA"]
        letters_no_expand, owners_no_expand = initials_seq(tokens, expand_allcaps=False)
        assert letters_no_expand == ["M", "R"]
        assert owners_no_expand == [0, 1]

        letters_expand, owners_expand = initials_seq(tokens, expand_allcaps=True)
        assert letters_expand == ["M", "R", "N", "A"]
        assert owners_expand == [0, 1, 1, 1]


def test_is_mixed_case_acronym():
    assert is_mixed_case_acronym("mRNA") is True
    assert is_mixed_case_acronym("PDF") is False
    assert is_mixed_case_acronym("rna") is False
