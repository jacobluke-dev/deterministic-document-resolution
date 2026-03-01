"use client";

import React from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import type { ResolveItem } from "@/lib/api/types";

function clamp01(v: number) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function sliceSafe(text: string, start: number, end: number) {
  if (start < 0 || end < 0 || start > text.length || end > text.length || end < start) return "";
  return text.slice(start, end);
}

export function ResultsTable({
  text,
  items,
  onSelectOccurrence,
  onCopyRow,
}: {
  text: string;
  items: ResolveItem[];
  onSelectOccurrence: (o: { start: number; end: number }) => void;
  onCopyRow: (row: ResolveItem) => void;
}) {
  const [openKey, setOpenKey] = React.useState<string | null>(null);

  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-gray-50 text-xs text-gray-700">
          <tr>
            <th className="px-3 py-2">Acronym</th>
            <th className="px-3 py-2">Definition</th>
            <th className="px-3 py-2">Offsets</th>
            <th className="px-3 py-2">Confidence</th>
            <th className="px-3 py-2">Source</th>
            <th className="px-3 py-2 text-right">Actions</th>
          </tr>
        </thead>

        <tbody>
          {items.map((row) => {
            const key = `${row.acronym}:${row.start}:${row.end}`;
            const isOpen = openKey === key;

            const snippet = sliceSafe(text, row.start, row.end);
            const conf = clamp01(row.confidence ?? 0);

            return (
              <React.Fragment key={key}>
                <tr className="border-t">
                  <td className="px-3 py-2 font-medium text-gray-900">
                    <button
                      className="hover:underline"
                      onClick={() => setOpenKey(isOpen ? null : key)}
                      aria-expanded={isOpen}
                    >
                      {row.acronym}
                    </button>
                  </td>
                  <td className="px-3 py-2 text-gray-800">{row.definition ?? "—"}</td>
                  <td className="px-3 py-2 text-gray-800">
                    <span title={`"${snippet}" (end-exclusive)`}>
                      {row.start}–{row.end}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <div className="w-24">
                        <ProgressBar value={conf} />
                      </div>
                      <span className="text-xs text-gray-600">{conf.toFixed(2)}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <Badge variant={row.source === "glossary" ? "accent" : "neutral"}>
                        {row.source ?? "—"}
                      </Badge>
                      {row.glossary_source ? (
                        <span className="text-xs text-gray-600" title={row.glossary_source}>
                          {row.glossary_source}
                        </span>
                      ) : null}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex justify-end gap-2">
                      {row.occurrences?.length ? (
                        <Button variant="secondary" onClick={() => setOpenKey(isOpen ? null : key)}>
                          {isOpen ? "Hide" : "Show"}
                        </Button>
                      ) : null}
                      <Button variant="secondary" onClick={() => onCopyRow(row)}>
                        Copy
                      </Button>
                    </div>
                  </td>
                </tr>

                {isOpen ? (
                  <tr className="border-t bg-gray-50">
                    <td colSpan={6} className="px-3 py-3">
                      <div className="text-xs text-gray-600 mb-2">Occurrences</div>
                      {row.occurrences?.length ? (
                        <div className="flex flex-wrap gap-2">
                          {row.occurrences.map((o, idx) => {
                            const snip = sliceSafe(text, o.start, o.end);
                            return (
                              <button
                                key={idx}
                                className="rounded border bg-white px-2 py-1 text-xs hover:bg-gray-100"
                                onClick={() => onSelectOccurrence({ start: o.start, end: o.end })}
                                title={`"${snip}" (end-exclusive)`}
                              >
                                {o.start}–{o.end}
                              </button>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-800">No occurrences provided.</div>
                      )}
                    </td>
                  </tr>
                ) : null}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
