import type { AcronymBlock } from "@/lib/api/types";

export type ResolveSense = {
  definition: string;
  confidence: number | null;
  source: "extracted" | "glossary";
};

export type ResolveRow = {
  acronym: string;
  definition: string | null;      // primary
  start: number;
  end: number;
  confidence: number | null;       // primary
  source: "extracted" | "glossary" | "—";
  glossaryLabel: string | null;
  occurrences: { start: number; end: number }[];

  senses: ResolveSense[];          // all senses (extracted + glossary)
};


export function toResolveRows(blocks: AcronymBlock[]): ResolveRow[] {
  return blocks.map((b) => {
    const extracted = (b.definitions ?? []).map((d) => ({
      definition: d.text,
      confidence: d.confidence ?? null,
      source: d.source, // "extracted" | "glossary" (per your Definition model)
    }));

    const glossary = (b.glossary?.matches ?? []).map((m) => ({
      definition: m.definition,
      confidence: m.confidence ?? null,
      source: "glossary" as const,
    }));

    // deterministic order: extracted ranked first (already ranked), then glossary ranked
    const senses = [...extracted, ...glossary];

    const primary = senses[0] ?? null;

    const source: ResolveRow["source"] = primary ? primary.source : "—";
    const glossaryLabel = glossary.length ? "glossary" : null;

    return {
      acronym: b.acronym,
      start: b.first_occurrence.start,
      end: b.first_occurrence.end,
      definition: primary?.definition ?? null,
      confidence: primary?.confidence ?? null,
      source,
      glossaryLabel,
      occurrences: (b.occurrences ?? []) as { start: number; end: number }[],
      senses,
    };
  });
}
