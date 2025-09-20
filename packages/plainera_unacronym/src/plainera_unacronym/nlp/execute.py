import json
from dataclasses import asdict
from pathlib import Path

from plainera_unacronym.nlp.detection.detector import Detector
from plainera_unacronym.nlp.common.types import SCHEMA_VERSION, DetectorConfig, DetectorResult


def _serialize(result: DetectorResult, *, pretty: bool = False) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "unique_acronyms": {k: asdict(v) for k, v in result.unique_acronyms.items()},
        "occurrences": [asdict(o) for o in result.occurrences],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def run_detection(
    text: str,
    *,
    parallel: bool = False,
    caps_ratio: float = 0.7,
    pretty: bool = False,
    as_json: bool = True,
    debug_reasons: bool = False,
    enable_dotted: bool = False,
) -> str | DetectorResult:
    cfg = DetectorConfig(
        require_caps_ratio=caps_ratio,
        debug_reasons=debug_reasons,
        enable_dotted=enable_dotted,
    )
    det = Detector(cfg)
    result = det.detect_parallel(text) if parallel else det.detect(text)
    return _serialize(result, pretty=pretty) if as_json else result


def execute_acronym_locator(
    file_path: str,
    *,
    parallel: bool = False,
    caps_ratio: float = 0.7,
    pretty: bool = False,
    as_json: bool = True,
    debug_reasons: bool = False,
    enable_dotted: bool = False,
) -> str | DetectorResult:
    """
    Run the acronym detector on the contents of a file and return results.

    This is a convenience wrapper around `run_detection(...)` that:
    1) reads `file_path` as UTF-8 text,
    2) invokes the detector with the given options, and
    3) returns either a JSON string (default) or a `DetectorResult` object.

    Args:
      file_path: Path to a UTF-8 text file to analyze.
      parallel: If True, may use a process pool for large inputs (same results as serial).
      caps_ratio: Required ratio of uppercase letters over letters for a token to be considered
        (digits ignored). Typical range: 0.6–0.8. Higher values are stricter.
      pretty: If True and `as_json=True`, pretty-prints the JSON with indentation.
      as_json: If True (default), return a JSON string. If False, return a `DetectorResult`.
      debug_reasons: If True, include a `reasons` list per occurrence explaining which
        heuristics fired and the effective threshold used (useful for logging/debugging).
      enable_dotted: If True, detect dotted initialisms like “U.S.” / “U.S.A.” and normalize
        their keys by stripping dots (e.g., “U.S.” → key “US”).

    Returns:
      str | DetectorResult:
        - If `as_json=True`: a JSON string with keys `schema_version`, `unique_acronyms`,
          and `occurrences`. Each occurrence includes `acronym`, offsets, confidence,
          `context_window`, `normalized_key`, and (optionally) `reasons`.
        - If `as_json=False`: a `DetectorResult` dataclass with the same information.

    Raises:
      FileNotFoundError: If `file_path` does not exist.
      PermissionError: If the file cannot be read due to permissions.
      UnicodeDecodeError: If the file is not valid UTF-8.
      OSError: For other I/O errors encountered while reading the file.

    Notes:
      - Offsets are Python code-point indices into the original text.
      - Parallel mode preserves determinism and first-occurrence selection.
      - Normalized keys canonicalize separators (e.g., “R & D” → “R&D”); when
        `enable_dotted=True`, dots are removed (“U.S.” → “US”).

    Examples:
      >>> # Return pretty JSON (good for logging or API responses)
      >>> json_str = execute_acronym_locator(
      ...     "sample.txt",
      ...     parallel=True,
      ...     debug_reasons=True,
      ...     enable_dotted=False,
      ...     pretty=True,
      ...     as_json=True,
      ... )
      >>> print(json_str[:120])

      >>> # Return structured result (good for programmatic use)
      >>> result = execute_acronym_locator("sample.txt", as_json=False)
      >>> list(result.unique_acronyms.keys())
      ['NHS', 'IT', 'R&D']
    """
    text = Path(file_path).read_text(encoding="utf-8")
    return run_detection(
        text,
        parallel=parallel,
        caps_ratio=caps_ratio,
        pretty=pretty,
        as_json=as_json,
        debug_reasons=debug_reasons,
        enable_dotted=enable_dotted,
    )
