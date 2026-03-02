"use client";

import React from "react";
import toast, {Toaster} from "react-hot-toast";

import {Panel} from "@/components/ui/Panel";
import {Button} from "@/components/ui/Button";
import {Skeleton} from "@/components/ui/Skeleton";
import {Toggle} from "@/components/ui/Toggle";
import {Modal} from "@/components/ui/Modal";

import {resolveText} from "@/lib/api/client";
import type {ResolveRequest} from "@/lib/api/types";
import {ResultsTable} from "@/components/demo/ResultsTable";
import FormTextarea from "@/components/form/FormTextArea";
import {ResolveRow, toResolveRows} from "@/lib/api/mapper";

const LS_KEY = "unacronym.demo.text";
const LS_REMEMBER = "unacronym.demo.remember";


export default function DemoPage() {

  type UiState =
    | { kind: "idle" }
    | { kind: "loading" }
    | { kind: "success"; rows: ResolveRow[] }
    | { kind: "error"; message: string; technical?: unknown };

  const [remember, setRemember] = React.useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(LS_REMEMBER) === "1";
  });

  const [text, setText] = React.useState<string>(() => {
    if (typeof window === "undefined") return "";
    const r = window.localStorage.getItem(LS_REMEMBER) === "1";
    return r ? window.localStorage.getItem(LS_KEY) ?? "" : "";
  });


  const [apiKey, setApiKey] = React.useState("");
  const [useApiKey, setUseApiKey] = React.useState(false);

  const [ui, setUi] = React.useState<UiState>({kind: "idle"});
  const isLocal = process.env.NEXT_PUBLIC_ENV === "local";

  const [showTech, setShowTech] = React.useState(false);
  const [techDetails, setTechDetails] = React.useState<unknown>(null);

  const [selected, setSelected] = React.useState<{ start: number; end: number } | null>(null);

  const abortRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LS_REMEMBER, remember ? "1" : "0");
    if (remember) window.localStorage.setItem(LS_KEY, text);
    if (!remember) window.localStorage.removeItem(LS_KEY);
  }, [remember, text]);

  const charCount = text.length;
  const tooLarge = charCount > 100_000;

  async function onResolve() {
    const trimmed = text.trim();

    if (!trimmed) {
      toast.error("Please paste some text.");
      setUi({kind: "error", message: "Please paste some text."});
      return;
    }

    if (tooLarge) {
      toast.error("Input too large (limit: 100,000 characters).");
      setUi({kind: "error", message: "Input too large (limit: 100,000 characters)."});
      return;
    }

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setSelected(null);
    setUi({kind: "loading"});

    const req: ResolveRequest = {
      text: trimmed,
      options: {
        locale: "en-GB",
        return_occurrences: true,
        include_glossary_enrichment: true,
      },
    };

    try {
      const data = await resolveText(req, ac.signal, useApiKey ? apiKey : undefined);

      const rows = toResolveRows(data.acronyms ?? []).sort((a, b) => a.start - b.start);
      setUi({kind: "success", rows});
      toast.success(`${rows.length} acronym(s) found.`);

      const live = document.getElementById("results-live-region");
      if (live) live.textContent = `${rows.length} acronyms found.`;
    } catch (e: any) {
      const message = e?.message ?? "Request failed.";
      setUi({kind: "error", message, technical: e?.details ?? e});
      setTechDetails(e?.details ?? e);
      toast.error(message);
    }
  }

  function onLoadSample() {
    setText(
      "The GPU (Graphics Processing Unit) is used for parallel workloads. Modern CPUs (Central Processing Units) may also support SIMD (Single Instruction, Multiple Data).",
    );
  }

  function copyJson(value: unknown) {
    navigator.clipboard.writeText(JSON.stringify(value, null, 2));
    toast.success("Copied to clipboard.");
  }

  const resultsCount = ui.kind === "success" ? ui.rows.length : 0;

  return (
    <>
      <Toaster position="top-right"/>

      <div className="mx-auto max-w-8xl p-4">
        <div className="mb-4">
          <h1 className="text-xl font-semibold text-gray-100">Demo Page do not paste any confidential data in here please.</h1>
          <p className="text-sm text-white">
            Paste text → Resolve → inspect deterministic offsets & sources.
          </p>
        </div>
        { isLocal ?
        <div className="mb-3 flex flex-col gap-2 bg-white rounded-lg border">
          <label className="flex items-center gap-2 text-sm text-gray-800">
            <input
              type="checkbox"
              checked={useApiKey}
              onChange={(e) => setUseApiKey(e.target.checked)}
              className="h-4 w-4"
            />
            Provide API key (local/dev only)
          </label>

          {useApiKey ? (
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste API key…"
              className="w-full rounded-md border px-3 py-2 text-sm shadow-sm"
              autoComplete="off"
            />
          ) : null}
        </div> : null
        }
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel
            title="Input"
            subtitle="Up to 100,000 characters"
            right={
              <div className="flex items-center gap-3">
                <Toggle checked={remember} onChange={setRemember} label="Remember input"/>
                <Button variant="secondary" onClick={onLoadSample}>
                  Load sample
                </Button>
              </div>
            }
          >
            <FormTextarea
              id="demo-text"
              label="Text"
              rows={16}
              value={text}
              onChange={(e) => setText(e.target.value)}
              maxLength={100_000}
              error={tooLarge ? "Over limit" : undefined}
              placeholder="Paste your text here…"
            />

            <div className="mt-3 flex items-center gap-2">
              <Button onClick={onResolve} disabled={ui.kind === "loading" || !text.trim() || tooLarge}>
                {ui.kind === "loading" ? "Resolving…" : "Resolve"}
              </Button>

              <Button
                variant="secondary"
                onClick={() => {
                  abortRef.current?.abort();
                  setUi({kind: "idle"});
                }}
                disabled={ui.kind !== "loading"}
              >
                Cancel
              </Button>

              <div className="ml-auto flex items-center gap-2">
                <Button
                  variant="secondary"
                  onClick={() => {
                    if (ui.kind === "success") copyJson(ui.rows);
                  }}
                  disabled={ui.kind !== "success" || ui.rows.length === 0}
                >
                  Copy all
                </Button>

                {ui.kind === "error" ? (
                  <Button variant="secondary" onClick={() => setShowTech(true)}>
                    Show technical details
                  </Button>
                ) : null}
              </div>
            </div>

            {ui.kind === "error" ? (
              <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                {ui.message}
              </div>
            ) : null}
          </Panel>

          <Panel
            title="Results"
            subtitle={ui.kind === "loading" ? "Loading…" : ui.kind === "success" ? `${resultsCount} found` : "—"}
            right={<span id="results-live-region" className="sr-only" aria-live="polite"/>}
          >
            {ui.kind === "loading" ? (
              <div className="space-y-2">
                <Skeleton className="h-10 w-full"/>
                <Skeleton className="h-10 w-full"/>
                <Skeleton className="h-10 w-full"/>
              </div>
            ) : ui.kind === "success" && ui.rows.length === 0 ? (
              <div className="rounded-md border p-4 text-sm text-gray-800">
                <div className="font-medium">No acronyms found</div>
                <div className="mt-1 text-gray-600">Try a longer passage, or use “Load sample”.</div>
              </div>
            ) : ui.kind === "success" ? (
              <ResultsTable
                text={text}
                items={ui.rows}
                onSelectOccurrence={(o) => setSelected(o)}
                onCopyRow={(row) => copyJson(row)}
              />
            ) : (
              <div className="rounded-md border p-4 text-sm text-gray-600">
                Paste text and click Resolve.
              </div>
            )}

            {selected ? (
              <div className="mt-4 rounded-md border p-3">
                <div className="text-xs text-gray-600">Preview (end-exclusive offsets)</div>
                <pre className="mt-2 whitespace-pre-wrap text-sm text-gray-900">
                  {renderHighlightedPreview(text, selected.start, selected.end)}
                </pre>
              </div>
            ) : null}
          </Panel>
        </div>
      </div>

      <Modal open={showTech} title="Technical details" onClose={() => setShowTech(false)}>
        <pre className="whitespace-pre-wrap text-xs text-gray-900">
          {JSON.stringify(techDetails, null, 2)}
        </pre>
      </Modal>
    </>
  );
}

function renderHighlightedPreview(text: string, start: number, end: number) {
  const before = text.slice(0, start);
  const mid = text.slice(start, end);
  const after = text.slice(end);
  return `${before}[${mid}]${after}`;
}
