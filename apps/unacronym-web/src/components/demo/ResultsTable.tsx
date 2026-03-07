"use client";

import React from "react";
import {Badge} from "@/components/ui/Badge";
import {Button} from "@/components/ui/Button";
import {ProgressBar} from "@/components/ui/ProgressBar";
import type {ResolveRow} from "@/lib/api/mapper";

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
  items: ResolveRow[];
  onSelectOccurrence: (o: { start: number; end: number }) => void;
  onCopyRow: (row: ResolveRow) => void;
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
                <td className="px-3 py-2 text-gray-800">
                  <div className="flex items-center gap-2">
                    <span>{row.definition ?? "—"}</span>

                    {row.senses?.length > 1 ? (
                      <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-700">
        +{row.senses.length - 1}
      </span>
                    ) : null}
                  </div>
                </td>
                <td className="px-3 py-2 text-gray-800">
                    <span title={`"${snippet}" (end-exclusive)`}>
                      {row.start}–{row.end}
                    </span>
                </td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <div className="w-24">
                      <ProgressBar value={conf}/>
                    </div>
                    <span className="text-xs text-gray-600">{conf.toFixed(2)}</span>
                  </div>
                </td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <Badge variant={row.source === "glossary" ? "accent" : "neutral"}>
                      {row.source}
                    </Badge>
                    {row.glossaryLabel ? (
                      <span className="text-xs text-gray-600">{row.glossaryLabel}</span>
                    ) : null}
                  </div>
                </td>
                <td className="px-3 py-2 text-right">
                  <div className="flex justify-end gap-2">
                    {row.occurrences?.length ? (
                      <Button variant="secondary" onClick={() => setOpenKey(isOpen ? null : key)}>
                        {isOpen ? "Close" : "Details"}
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
                  <td colSpan={6} className="px-3 py-3 space-y-4">
                    {/* Senses / Meaning */}
                    {row.senses?.length ? (
                      <div>
                        <div className="text-md text-gray-600 mb-2">
                          Meaning {row.senses.length > 1 ? `(${row.senses.length})` : ""}
                        </div>

                        <div className="space-y-2">
                          {row.senses.map((s, idx) => {
                            const conf2 = clamp01(s.confidence ?? 0);

                            return (
                              <div
                                key={idx}
                                className="rounded border bg-white px-3 py-2 flex items-center justify-between gap-4"
                              >
                                <div className="min-w-0">
                                  <div className="text-sm text-gray-900 truncate">{s.definition}</div>
                                  <div className="mt-1 flex items-center gap-2">
                                    <Badge variant={s.source === "glossary" ? "accent" : "neutral"}>
                                      {s.source}
                                    </Badge>
                                    <span className="text-xs text-gray-600">{conf2.toFixed(2)}</span>
                                  </div>
                                </div>

                                <div className="w-28 shrink-0">
                                  <ProgressBar value={conf2}/>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : null}

                    {/* Occurrences */}
                    <div>
                      <div className="text-md text-gray-600 mb-2">Occurrences</div>
                      {row.occurrences?.length ? (
                        <div className="flex flex-wrap gap-2">
                          {row.occurrences.map((o, idx) => {
                            const snip = sliceSafe(text, o.start, o.end);
                            return (
                              <button
                                key={idx}
                                className="rounded border text-gray-700 bg-white px-2 py-1"
                                onClick={() => onSelectOccurrence({start: o.start, end: o.end})}
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
                    </div>
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
