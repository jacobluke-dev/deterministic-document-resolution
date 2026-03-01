import {AcronymBlock} from "@/lib/api/types";


export type ResolveRow = {
  acronym: string;
  definition: string | null;
  start: number;
  end: number;
  confidence: number | null; // 0..1
  source: "extracted" | "glossary" | "—";
  occurrences: { start: number; end: number }[];
  glossary_source: string | null; // keep for UI (e.g. "system" / domain)
};

export function toResolveRows(blocks: AcronymBlock[]): ResolveRow[] {
  return blocks.map((b) => {
    const best = (b.definitions ?? [])[0] ?? null;

    return {
      acronym: b.acronym,
      start: b.first_occurrence.start,
      end: b.first_occurrence.end,
      definition: best?.text ?? null,
      confidence: best?.confidence ?? null,
      source: best?.source ?? "—",
      occurrences: (b.occurrences ?? []) as { start: number; end: number }[],
      glossary_source: b.glossary?.matches?.[0]?.source ?? null,
    };
  });
}
