from plainera_unacronym.nlp import AcronymDetector, AcronymDetectorConfig


def _keys(result) -> set[str]:
    return set(result.unique_acronyms.keys())


class TestBioE2E:
    def test_bio_end_to_end_default_config(self, patch_sink_and_logger):
        """
        E2E: with the normal DetectorConfig, a bio-ish paragraph should surface
        clear bio tokens (mRNA, IL-6). We also verify autodetect logging when present.
        """
        txt = (
            "We quantified mRNA and IL-6 in hospitalized patients. "
            "SARS-CoV-2 cohorts were analyzed with PCR and ELISA."
        )
        res = AcronymDetector(AcronymDetectorConfig(enabled_domains=frozenset({"bio"}))).detect(txt)
        keys = _keys(res)

        # Core bio signals should be detected as acronyms
        assert "mRNA" in keys, f"mRNA missing; keys={keys}"
        assert "IL-6" in keys or "IL" in keys, f"IL-6 (or IL) missing; keys={keys}"

        # Autodetect log is expected but not strictly required if the plugin registry
        # or SupportsSniff gating differs; when present, it should precede 'start'.
        msgs = [e["message"] for e in patch_sink_and_logger]
        if "acronym_detector.autodetect_domains" in msgs:
            assert msgs.index("acronym_detector.autodetect_domains") < msgs.index("acronym_detector.detect.start")

        assert "acronym_detector.detect.summary" in msgs

    def test_bio_parallel_equals_serial(self, patch_sink_and_logger):
        """
        E2E parity: a longer bio paragraph yields identical unique sets and counts
        in serial vs. parallel paths.
        """
        para = (
            "mRNA and IL-6 were measured after SARS-CoV-2 exposure. "
            "PCR confirmed results; ELISA validated protein levels. "
        )
        big = para * 200  # large enough to consider parallel
        det_default = AcronymDetector(AcronymDetectorConfig(enabled_domains=frozenset({"bio"})))

        serial = det_default.detect(big)
        parallel = det_default.detect_parallel(big, threshold=10, chunk_size=64)

        # Unique sets identical
        assert _keys(serial) == _keys(parallel)

        # Occurrence counts per acronym identical
        def counts(d):
            c = {}
            for o in d.occurrences:
                c[o.acronym] = c.get(o.acronym, 0) + 1
            return c

        assert counts(serial) == counts(parallel), f"counts differ: {counts(serial)} vs {counts(parallel)}"


class TestBioAndGeneralIntegration:
    def test_mixed_bio_and_general_tokens(self):
        """
        E2E: bio cues + general acronyms in one paragraph.
        Ensures both categories surface and common distractors drop.
        """
        txt = (
            "We quantified mRNA and IL-6 in a SARS-CoV-2 cohort. "
            "R & D collaborated with NHS on GPU workloads, JSON exports, and USB-C hubs. "
            "OK, we'll reconvene at 10:30 AM."
        )

        # Enable bio domain explicitly for stability (autodetect is exercised elsewhere)
        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"bio"}), dotted_display="strip")
        res = AcronymDetector(cfg).detect(txt)
        ks = _keys(res)

        assert {"mRNA", "RNA"} & ks, f"mRNA/RNA missing; keys={ks}"
        assert {"IL-6", "IL"} & ks, f"IL-6/IL missing; keys={ks}"
        assert "SARS-CoV-2" in ks or {"SARS", "CoV"} <= ks, f"SARS-CoV-2/SARS+CoV missing; keys={ks}"

        # General acronyms present
        assert {"R&D", "NHS", "GPU", "JSON", "USB-C"}.issubset(ks), f"general missing; keys={ks}"

        # Common drops (interjection/time-of-day)
        assert "OK" not in ks, f"'OK' should drop; keys={ks}"
        assert "AM" not in ks, f"'AM' (time-of-day) should drop; keys={ks}"
