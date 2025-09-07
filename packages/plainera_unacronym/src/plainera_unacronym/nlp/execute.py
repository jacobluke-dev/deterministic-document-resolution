import argparse, json, sys
from dataclasses import asdict
from src.plainera_unacronym.nlp.detector import Detector
from src.plainera_unacronym.nlp.types import DetectorConfig


def execute() -> None:
    p = argparse.ArgumentParser(description="Detect acronyms from stdin or file.")
    p.add_argument("file", nargs="?", help="Input file (defaults to stdin).")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--parallel", action="store_true", help="Use process pool for large inputs.")
    p.add_argument("--caps-ratio", type=float, default=0.7)
    args = p.parse_args()

    text = sys.stdin.read() if not args.file else open(args.file, "r", encoding="utf-8").read()
    cfg = DetectorConfig(require_caps_ratio=args.caps_ratio)
    det = Detector(cfg)

    if args.parallel:
        result = det.detect_parallel(text)
    else:
        result = det.detect(text)

    payload = {
        "unique_acronyms": {k: asdict(v) for k, v in result.unique_acronyms.items()},
        "occurrences": [asdict(o) for o in result.occurrences],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
