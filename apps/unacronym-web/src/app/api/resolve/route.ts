import { NextResponse } from "next/server";
import type { ResolveRequest, ResolveResponse } from "@/lib/api/types";

export const runtime = "nodejs"; // keep it node for env + SDK compatibility (edge later if you want)

function jsonError(status: number, message: string, details?: unknown) {
  return NextResponse.json({ message, details }, { status });
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

  const upstreamBase = process.env.UNACRONYM_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL; // prefer server-only var
  const apiKey = process.env.UNACRONYM_API_KEY;

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
    // If you want to use your generated OpenAPI SDK, do it here instead.
    // Keeping it plain fetch avoids SDK coupling while UN-21 stabilises.
    const resp = await fetch(`${upstreamBase.replace(/\/$/, "")}/v1/resolve`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify(payload),
    });

    if (resp.status === 401) return jsonError(401, "API key required/invalid.");
    if (resp.status === 413) return jsonError(413, "Input too large (request limit exceeded).");
    if (resp.status === 422) return jsonError(422, "Please paste some text.");
    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      return jsonError(resp.status, "Request failed.", txt || undefined);
    }

    const data = (await resp.json()) as ResolveResponse;
    return NextResponse.json(data);
  } catch (e) {
    return jsonError(0, "Network error: could not reach the API.", String(e));
  }
}
