import type { AcronymBlock } from "@/lib/api/types";

export type ResolveRow = {
  acronym: string;
  definition: string | null;
  start: number;
  end: number;
  confidence: number | null;
  source: "extracted" | "glossary" | "—";
  glossaryLabel: string | null; // shown only when glossary contributed something
  occurrences: { start: number; end: number }[];
};

export function toResolveRows(blocks: AcronymBlock[]): ResolveRow[] {
  return blocks.map((b) => {
    const extractedBest = (b.definitions ?? [])[0] ?? null;
    const glossaryBest = b.glossary?.matches?.[0] ?? null;

    // Prefer extracted definition; fall back to glossary definition if present
    const definition = extractedBest?.text ?? glossaryBest?.definition ?? null;

    // Confidence: extracted first; else glossary match confidence; else null
    const confidence = extractedBest?.confidence ?? glossaryBest?.confidence ?? null;

    // Source column: extracted if extracted exists, otherwise glossary if glossary exists
    const source: ResolveRow["source"] = extractedBest
      ? extractedBest.source // "extracted" | "glossary"
      : glossaryBest
        ? "glossary"
        : "—";

    // Glossary label: only show if there is a glossary match (don’t show "system")
    const glossaryLabel = glossaryBest ? "glossary" : null;

    return {
      acronym: b.acronym,
      start: b.first_occurrence.start,
      end: b.first_occurrence.end,
      definition,
      confidence,
      source,
      glossaryLabel,
      occurrences: (b.occurrences ?? []) as { start: number; end: number }[],
    };
  });
}
