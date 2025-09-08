import json
from dataclasses import asdict
from pathlib import Path

from src.plainera_unacronym.nlp.detector import Detector
from src.plainera_unacronym.nlp.types import DetectorConfig, DetectorResult

def _serialize(result: DetectorResult, *, pretty: bool = False) -> str:
    payload = {
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
) -> str | DetectorResult:
    cfg = DetectorConfig(require_caps_ratio=caps_ratio)
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
) -> str | DetectorResult:
    """Convenience wrapper that reads a file, then delegates to run_detection."""
    text = Path(file_path).read_text(encoding="utf-8")
    return run_detection(text, parallel=parallel, caps_ratio=caps_ratio, pretty=pretty, as_json=as_json)
