export type ResolveOptions = {
  locale?: string; // "en-GB"
  return_occurrences?: boolean;
  include_glossary_enrichment?: boolean;
};

export type ResolveRequest = {
  text: string;
  options?: ResolveOptions;
};

// Adjust these to match your OpenAPI schema
export type ResolveOccurrence = {
  start: number;
  end: number; // end-exclusive
  // optional fields depending on API
  context_left?: string;
  context_right?: string;
};

export type ResolveItem = {
  acronym: string;
  definition?: string | null;
  confidence?: number | null; // 0..1
  source?: "extracted" | "glossary" | string;
  // “primary” first occurrence
  start: number;
  end: number; // end-exclusive
  occurrences?: ResolveOccurrence[];
  glossary_source?: string | null; // e.g., "Internal glossary v3" / doc id / citation pointer
};

export type ResolveResponse = {
  items: ResolveItem[];
};
