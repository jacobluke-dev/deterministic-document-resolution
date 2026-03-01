// src/lib/api/types.ts

export type ResolveOptions = {
  locale?: "en-GB" | "en-US";
  window_chars?: number;
  max_definitions_per_acronym?: number;
  include_glossary_enrichment?: boolean;
  return_occurrences?: boolean;
  min_confidence?: number;
};

export type ResolveRequest = {
  text: string;
  options?: ResolveOptions | null;
};

export type Span = { start: number; end: number };

export type DefinitionSource = "extracted" | "glossary";

export type Definition = {
  text: string;
  start: number | null;
  end: number | null;
  confidence: number; // 0..1
  source: DefinitionSource;
};

export type GlossaryMatch = {
  definition: string;
  domain: string | null;
  lang: string;
  confidence: number;
  source?: "system";
};

export type GlossaryBlock = {
  matches: GlossaryMatch[];
};

export type AcronymBlock = {
  acronym: string;
  first_occurrence: Span;
  definitions?: Definition[]; // API says required in schema examples, but be defensive
  occurrences?: Span[] | null;
  glossary?: GlossaryBlock | null;
};

export type ResolveMeta = {
  processing_ms: number;
  model_version: string;
  input_chars: number;
};

export type ResolveResponse = {
  acronyms: AcronymBlock[];
  meta: ResolveMeta;
};
