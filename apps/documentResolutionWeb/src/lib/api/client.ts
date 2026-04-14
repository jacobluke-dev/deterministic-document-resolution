import type {ResolveRequest, ResolveResponse} from "@/lib/api/types";
import {ErrorEnvelope, isErrorEnvelope, ResolveClientError} from "@/utils/errors";


export async function resolveText(
  req: ResolveRequest,
  signal?: AbortSignal,
  apiKeyOverride?: string,
): Promise<ResolveResponse> {
  const headers: Record<string, string> = {"content-type": "application/json"};

  if (apiKeyOverride?.trim()) {
    headers["x-unacronym-api-key"] = apiKeyOverride.trim();
  }

  let resp: Response;
  try {
    resp = await fetch("/api/resolve", {
      method: "POST",
      headers,
      body: JSON.stringify(req),
      signal,
    });
  } catch (e) {
    // network error / CORS / proxy died
    const err: ResolveClientError = {
      status: 0,
      message: "Network error: could not reach the server.",
      details: e instanceof Error ? e.message : String(e),
    };
    throw err;
  }

  if (!resp.ok) {
    const contentType = resp.headers.get("content-type") ?? "";
    const payload =
      contentType.includes("application/json")
        ? await resp.json().catch(() => null)
        : await resp.text().catch(() => null);

    const message = (() => {
      if (!isErrorEnvelope(payload)) return "Request failed.";
      const m = (payload as ErrorEnvelope).message;
      return typeof m === "string" && m.trim() ? m : "Request failed.";
    })();

    const details = (() => {
      if (!isErrorEnvelope(payload)) return payload || undefined;
      const d = (payload as ErrorEnvelope).details;
      return d ?? payload ?? undefined;
    })();

    const err: ResolveClientError = {status: resp.status, message, details};
    throw err;
  }

  return (await resp.json()) as ResolveResponse;
}
