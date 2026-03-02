import { describe, it, expect } from "vitest";
import {AcronymBlock} from "@/lib/api/types";
import {toResolveRows} from "@/lib/api/mapper";


describe("toResolveRows", () => {
  it("prefers extracted definition, falls back to glossary", () => {
    const blocks: AcronymBlock[] = [
      {
        acronym: "GPU",
        first_occurrence: { start: 4, end: 7 },
        definitions: [
          { text: "Graphics Processing Unit", start: 9, end: 33, confidence: 0.95, source: "extracted" },
        ],
        occurrences: [{ start: 4, end: 7 }],
        glossary: { matches: [{ definition: "Graphics Processing Unit.", domain: null, lang: "en", confidence: 1, source: "system" }] },
      },
      {
        acronym: "NHS",
        first_occurrence: { start: 0, end: 3 },
        definitions: [],
        occurrences: [{ start: 0, end: 3 }],
        glossary: { matches: [{ definition: "National Health Service.", domain: null, lang: "en", confidence: 1, source: "system" }] },
      },
    ];

    const rows = toResolveRows(blocks);

    expect(rows[0].definition).toBe("Graphics Processing Unit");
    expect(rows[0].source).toBe("extracted");
    expect(rows[0].glossaryLabel).toBe("glossary");

    expect(rows[1].definition).toBe("National Health Service.");
    expect(rows[1].source).toBe("glossary");
  });
});
