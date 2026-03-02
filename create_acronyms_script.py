#!/usr/bin/env python3
"""
Generate a .txt file containing unique acronyms + dummy definitions until a target
character count is reached.

Install: nothing extra (stdlib only)

Run:
  python gen_acronyms_txt.py --out acronyms_100k.txt --target-chars 100000 --seed 123
"""

from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class GenConfig:
    target_chars: int = 100_000
    min_acr_len: int = 2
    max_acr_len: int = 8
    words_min: int = 3
    words_max: int = 9


_WORD_BANK_A = [
    "Adaptive", "Advanced", "Atomic", "Augmented", "Autonomous", "Asynchronous",
    "Binary", "Boundary", "Bayesian", "Buffered", "Concurrent", "Cryptographic",
    "Distributed", "Deterministic", "Dynamic", "Elastic", "Embedded", "Encrypted",
    "Federated", "Fault-tolerant", "Granular", "Hybrid", "Incremental", "Integrated",
    "Linear", "Modular", "Neural", "Optimised", "Parallel", "Predictive",
    "Quantum", "Recursive", "Robust", "Scalable", "Semantic", "Sparse",
    "Streaming", "Structured", "Unified", "Verified",
]
_WORD_BANK_B = [
    "Access", "Alignment", "Allocator", "Anchor", "Boundary", "Bridge", "Cache",
    "Classifier", "Controller", "Decoder", "Detector", "Engine", "Extractor",
    "Filter", "Gateway", "Indexer", "Kernel", "Ledger", "Mapper", "Model",
    "Monitor", "Pipeline", "Processor", "Protocol", "Resolver", "Router",
    "Scheduler", "Service", "Signature", "Tokenizer", "Validator", "Worker",
]


def _initial_letter(word: str) -> str | None:
    """
    Return the first A-Z letter in `word` (uppercased), or None if no letter.
    Handles tokens like "Fault-tolerant", "v11", "id:123" safely.
    """
    for ch in word:
        if ch.isalpha():
            return ch.upper()
    return None


def _mk_entry_words_and_acronym(
    rng: random.Random,
    *,
    words_min: int,
    words_max: int,
    expand_words: int = 5,
) -> tuple[list[str], str]:
    """
    Build a word list and an acronym that matches the initials of the first
    `expand_words` words.

    Returns:
        (words, acronym)
    """
    n = rng.randint(words_min, words_max)
    words: list[str] = []

    # Generate definition words (no version suffixes in the expansion region)
    for i in range(n):
        bank = _WORD_BANK_A if rng.random() < 0.55 else _WORD_BANK_B
        w = rng.choice(bank)

        # Keep the first `expand_words` clean so initials are stable.
        if i >= expand_words and rng.random() < 0.08:
            w = f"{w} v{rng.randint(2, 12)}"

        words.append(w)

    k = max(1, min(expand_words, len(words)))
    initials = []
    for w in words[:k]:
        init = _initial_letter(w)
        if init is None:
            # Very unlikely with your banks, but fallback to a random letter.
            init = rng.choice(string.ascii_uppercase)
        initials.append(init)

    acr = "".join(initials)
    # Insert the acronym token immediately after the expansion phrase
    words.insert(k, f"({acr})")
    return words, acr


def _mk_acronym(rng: random.Random, min_len: int, max_len: int) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def _mk_definition_with_inline_acr(
    rng: random.Random,
    acr: str,
    words_min: int,
    words_max: int,
    *,
    insert_after_words: int = 5,
) -> str:
    """
    Build a definition and insert the acronym inline after N leading words.

    Example:
      Boundary Modular Detector Robust Federated (BTTF) Model Predictive Access
    """
    n = rng.randint(words_min, words_max)
    words: list[str] = []
    for _ in range(n):
        bank = _WORD_BANK_A if rng.random() < 0.55 else _WORD_BANK_B
        w = rng.choice(bank)
        if rng.random() < 0.08:
            w = f"{w} v{rng.randint(2, 12)}"
        words.append(w)

    # Ensure the insertion point is valid.
    k = max(1, min(insert_after_words, len(words)))
    words.insert(k, f"({acr})")

    tail = ""
    if rng.random() < 0.20:
        tail = f" ({rng.choice(['RFC', 'Spec', 'Draft', 'Annex'])}-{rng.randint(1, 99)})"
    if rng.random() < 0.12:
        tail += f" — id:{rng.randint(10_000, 999_999)}"

    return " ".join(words) + tail


def write_unique_acronyms_txt(out_path: str, cfg: GenConfig, *, seed: int | None = None) -> int:
    """
    Writes lines of "ACR — Definition" to `out_path` until `cfg.target_chars` is met.

    Returns:
        Approx character count written (based on text content length).
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    chars = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        while chars < cfg.target_chars:
            words, acr = _mk_entry_words_and_acronym(
                rng,
                words_min=cfg.words_min,
                words_max=cfg.words_max,
                expand_words=5,  # <- controls acronym length + where (ACR) is inserted
            )

            if acr in seen:
                continue
            seen.add(acr)

            tail = ""
            if rng.random() < 0.20:
                tail = f" ({rng.choice(['RFC', 'Spec', 'Draft', 'Annex'])}-{rng.randint(1, 99)})"
            if rng.random() < 0.12:
                tail += f" — id:{rng.randint(10_000, 999_999)}"

            line = f"{' '.join(words)}{tail}\n"
            f.write(line)
            chars += len(line)

    return chars


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a TXT full of unique acronyms for pressure testing.")
    p.add_argument("--out", default="acronyms.txt", help="Output TXT path")
    p.add_argument("--target-chars", type=int, default=100_000, help="Target character count (approx)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for repeatability")
    p.add_argument("--min-acr-len", type=int, default=2, help="Minimum acronym length")
    p.add_argument("--max-acr-len", type=int, default=8, help="Maximum acronym length")
    args = p.parse_args()

    cfg = GenConfig(
        target_chars=args.target_chars,
        min_acr_len=args.min_acr_len,
        max_acr_len=args.max_acr_len,
    )
    chars = write_unique_acronyms_txt(args.out, cfg, seed=args.seed)

    print(f"Wrote {args.out}")
    print(f"Approx chars: {chars:,}")
    print("Text printer go brrrr.")


if __name__ == "__main__":
    main()
