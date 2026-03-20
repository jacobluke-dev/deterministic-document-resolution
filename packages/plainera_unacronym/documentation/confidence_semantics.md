# Architecture Decision Record, ADR: Confidence semantics in extraction + meaning resolution

## Context
We compute confidence values for extracted acronym definitions across multiple strategies, on
detection, and extraction (anchored, harvest, backref, global). Previously, these values
existed but did not have a clear, consistent effect on downstream decisions.

## Decision
Confidence is treated as a deterministic *ranking signal* among structurally valid candidates.

- Confidence is a scalar in **[0.0, 0.99]** (clamped).
- It is **not** a calibrated probability.
- Structural gates determine whether a candidate is allowed.
- Confidence determines which allowed candidate wins when candidates compete.

## Rules
1. **No confidence-only rejection**
   - A structurally valid candidate is never dropped solely due to low confidence.

2. **Winner selection / dedupe**
   - When two definitions collapse to the same meaning key, the winner is the one with higher
     `definition_confidence`. Tie-break remains deterministic (first-seen unless otherwise stated).

3. **Meaning-level confidence**
   - A meaning carries `meaning_confidence` equal to the maximum `definition_confidence` among merged
     supports for that meaning.

4. **Disambiguation tie-break**
   - Occurrence resolution uses distance/overlap scoring as primary signals.
   - Confidence may be used only as a *small* tie-breaker (explicit, deterministic), never as the
     dominant scoring signal.

## Observability
Confidence and its reasons must be observable in debug output / traces and stable across runs
given identical inputs.
