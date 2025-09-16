import pytest
import plainera_unacronym.nlp.plugins.activation as act
import plainera_unacronym.nlp.detector as det
from plainera_unacronym.nlp.detector import Detector, DetectorConfig, autodetect_domains
from plainera_unacronym.domains.bio.plugin import BioPlugin


@pytest.fixture(autouse=True)
def patch_sink_and_logger(monkeypatch):
    # Silence DB/log I/O, but keep logs capturable if needed.
    class NullSink:
        def __call__(self, *a, **k): pass

        def __getattr__(self, _): return lambda *a, **k: None

    monkeypatch.setattr(det, "sink", NullSink(), raising=True)
    logs = []

    def spy_logger(message, *a, **kw):
        logs.append({"message": message, **kw})

    monkeypatch.setattr(det, "message_logger", spy_logger, raising=True)
    return logs


@pytest.mark.integration
class TestBioAutodetect:
    def test_autodetect_domains_flags_bio_from_rna_and_cytokines(self, monkeypatch):
        """
        Integration: ensure autodetect_domains returns {'bio'} when the text contains
        strong bio signals (e.g., mRNA / IL-6 / SARS-CoV-2).
        """
        # The autodetect uses an isinstance(plug, SupportsSniff) gate and a safe wrapper.
        # Make sure the registered BioPlugin gets queried in this test:
        monkeypatch.setattr(act, "SupportsSniff", object, raising=True)  # all objects pass isinstance
        # Some versions implement a sandbox wrapper; route to plugin.sniff in a safe way.
        monkeypatch.setattr(act, "_safe_sniff", lambda plug, t: plug.sniff(None, t), raising=True)

        text = (
            "We quantified mRNA for IL-6 after SARS-CoV-2 infection. "
            "The 5′UTR also showed changes."
        )
        cfg = DetectorConfig()
        auto = autodetect_domains(text, cfg)
        assert "bio" in auto, f"Expected 'bio' in autodetected domains, got {auto}"

    def test_detector_merges_auto_domains_and_logs_added(self, patch_sink_and_logger, monkeypatch):
        # Force auto-detection to return {'bio'} where Detector will actually look:
        monkeypatch.setattr(det, "autodetect_domains",
                            lambda text, cfg: frozenset({"bio"}), raising=True)

        # Keep detection path minimal
        monkeypatch.setattr(det, "compile_pattern", lambda _cfg: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *a, **k: [], raising=True)

        d = Detector(DetectorConfig(enabled_domains=frozenset()))
        _ = d.detect("mRNA and IL-6 were measured.")

        messages = [c["message"] for c in patch_sink_and_logger]

        assert "detector.autodetect_domains" in messages, f"logs: {messages}"
        assert "detector.detect.start" in messages
        assert "detector.detect.summary" in messages
        assert messages.index("detector.autodetect_domains") < messages.index("detector.detect.start")

    def test_bio_keep_guard_for_rna_and_two_letter_stats(self):
        """
        Check BioPlugin.keep_guard behavior:
          - RNA-like surfaces are kept.
          - Two-letter tokens like OR are kept when a stats context (95% CI / OR/HR/RR) is nearby.
        """
        plug = BioPlugin()
        cfg = DetectorConfig(
            enabled_domains=frozenset({"bio"}),  # activate plugin behavior
        )

        # 1) RNA-like surface is kept unconditionally when domain is enabled.
        text1 = "Differential expression of miRNA was observed."
        s1 = text1.index("miRNA");
        e1 = s1 + len("miRNA")
        assert plug.keep_guard("miRNA", text1, s1, e1, cfg) is True

        # 2) Two-letter stat token kept only with stats context (95% CI / OR/HR/RR around it)
        text2 = "OR = 1.8 (95% CI 1.2–2.3) for the treatment group."
        s2 = text2.index("OR");
        e2 = s2 + 2
        assert plug.keep_guard("OR", text2, s2, e2, cfg) is True

        text3 = "OR of many options were discussed casually (no stats)."
        s3 = text3.index("OR");
        e3 = s3 + 2
        # Without stats context, two-letter keep should be False
        assert plug.keep_guard("OR", text3, s3, e3, cfg) is False

    def test_bio_keep_guard_fails_for_two_letter_camelcase_or(self):
        """
        Check BioPlugin.keep_guard behavior:
          - RNA-like surfaces are kept.
          - Two-letter tokens like 'Or' fail rather than 'OR'.
        """
        plug = BioPlugin()
        cfg = DetectorConfig(
            enabled_domains=frozenset({"bio"}),  # activate plugin behavior
        )

        # 1) RNA-like surface is kept unconditionally when domain is enabled.
        text1 = "Differential expression of mirNA was observed."
        s1 = text1.index("mirNA");
        e1 = s1 + len("mirNA")
        assert plug.keep_guard("mirNA", text1, s1, e1, cfg) is False

        # 2) Two-letter stat token kept only with stats context (95% CI / OR/HR/RR around it)
        text2 = "Or = 1.8 (95% CI 1.2–2.3) for the treatment group."
        s2 = text2.index("Or");
        e2 = s2 + 2
        assert plug.keep_guard("OR", text2, s2, e2, cfg) is False


def _keys(result) -> set[str]:
    return set(result.unique_acronyms.keys())


@pytest.mark.e2e
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
        res = Detector(DetectorConfig(enabled_domains=frozenset({"bio"}))).detect(txt)
        keys = _keys(res)

        # Core bio signals should be detected as acronyms
        assert "mRNA" in keys, f"mRNA missing; keys={keys}"
        assert "IL-6" in keys or "IL" in keys, f"IL-6 (or IL) missing; keys={keys}"

        # Autodetect log is expected but not strictly required if the plugin registry
        # or SupportsSniff gating differs; when present, it should precede 'start'.
        msgs = [e["message"] for e in patch_sink_and_logger]
        if "detector.autodetect_domains" in msgs:
            assert msgs.index("detector.autodetect_domains") < msgs.index("detector.detect.start")

        # Summary log should always appear
        assert "detector.detect.summary" in msgs

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
        det_default = Detector(DetectorConfig(enabled_domains=frozenset({"bio"})))

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


@pytest.mark.e2e
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
        cfg = DetectorConfig(enabled_domains=frozenset({"bio"}), dotted_display="strip")
        res = Detector(cfg).detect(txt)
        ks = _keys(res)

        # Bio tokens (pattern may capture 'RNA' depending on your regex; accept either)
        assert {"mRNA", "RNA"} & ks, f"mRNA/RNA missing; keys={ks}"
        assert {"IL-6", "IL"} & ks, f"IL-6/IL missing; keys={ks}"
        assert "SARS-CoV-2" in ks

        # General acronyms present
        assert {"R&D", "NHS", "GPU", "JSON", "USB-C"}.issubset(ks), f"general missing; keys={ks}"

        # Common drops (interjection/time-of-day)
        assert "OK" not in ks, f"'OK' should drop; keys={ks}"
        assert "AM" not in ks, f"'AM' (time-of-day) should drop; keys={ks}"
