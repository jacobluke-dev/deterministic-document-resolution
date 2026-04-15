import {NextResponse} from "next/server";
import type {ResolveRequest, ResolveResponse} from "@/lib/api/types";

export const runtime = "nodejs"; // keep it node for env + SDK compatibility

function normaliseStatus(x: unknown, fallback = 502): number {
  const n = typeof x === "number" ? x : Number(x);
  return Number.isFinite(n) && n >= 200 && n <= 599 ? n : fallback;
}

function jsonError(status: unknown, message: string, details?: unknown) {
  return NextResponse.json({message, details}, {status: normaliseStatus(status)});
}

export async function POST(req: Request) {
  let body: ResolveRequest | null = null;

  try {
    body = (await req.json()) as ResolveRequest;
  } catch {
    return jsonError(400, "Invalid JSON body.");
  }

  const text = body?.text?.trim() ?? "";
  if (!text) return jsonError(422, "Please paste some text.");

  const upstreamBase = process.env.DOCUMENT_RESOLUTION_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL; // prefer server-only var

  const serverKey = process.env.DOCUMENT_RESOLUTION_API_KEY;
  const env = process.env.NEXT_PUBLIC_ENV ?? "local";

// Optional per-request override from demo UI
  const overrideKey = req.headers.get("x-document-resolution-api-key") ?? undefined;

  const apiKey =
    env === "local" && overrideKey?.trim()
      ? overrideKey.trim()
      : serverKey;

  if (!apiKey) {
    return NextResponse.json(
      {message: "Server misconfigured: missing API key."},
      {status: 500},
    );
  }


  if (!upstreamBase) return jsonError(500, "Server misconfigured: missing API base URL.");
  if (!apiKey) return jsonError(500, "Server misconfigured: missing API key.");

  const options = body.options ?? {};
  const payload: ResolveRequest = {
    text,
    options: {
      locale: options.locale ?? "en-GB",
      return_occurrences: options.return_occurrences ?? true,
      include_glossary_enrichment: options.include_glossary_enrichment ?? true,
    },
  };

  try {
    // If we want to use our generated OpenAPI SDK, do it here instead.
    // Keeping it plain fetch avoids SDK coupling while UN-21 stabilises.
    const resp = await fetch(`${upstreamBase.replace(/\/$/, "")}/v1/resolve`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify(payload),
    });

    if (resp.status === 401) return jsonError(401, "API key required/invalid.");
    if (resp.status === 413) return jsonError(413, "Input too large (request limit exceeded).");
    if (resp.status === 422) return jsonError(422, "Please paste some text.");
    if (!resp.ok) {
      const contentType = resp.headers.get("content-type") ?? "";
      let details: unknown = undefined;

      if (contentType.includes("application/json")) {
        details = await resp.json().catch(() => undefined);
      } else {
        const txt = await resp.text().catch(() => "");
        details = txt || undefined;
      }

      return jsonError(resp.status, "Request failed.", details);
    }

    // success
    const data = (await resp.json()) as ResolveResponse;
    return NextResponse.json(data, {status: resp.status});
  } catch (e) {
    return jsonError(502, "Network error: could not reach the API.", {
      upstreamBase,
      error: e instanceof Error ? e.message : String(e),
    });
  }
}
