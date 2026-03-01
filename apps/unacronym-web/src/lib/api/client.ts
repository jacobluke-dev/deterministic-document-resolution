import type { ResolveRequest, ResolveResponse } from "@/lib/api/types";

export async function resolveText(
  req: ResolveRequest,
  signal?: AbortSignal,
  apiKeyOverride?: string,
): Promise<ResolveResponse> {
  const headers: Record<string, string> = { "content-type": "application/json" };

  if (apiKeyOverride?.trim()) {
    headers["x-unacronym-api-key"] = apiKeyOverride.trim();
  }

  const resp = await fetch("/api/resolve", {
    method: "POST",
    headers,
    body: JSON.stringify(req),
    signal,
  });

  if (!resp.ok) {
    const payload = await resp.json().catch(() => null);
    const message = payload?.message ?? "Request failed.";
    const err = { status: resp.status, message, details: payload?.details };
    throw err;
  }

  return (await resp.json()) as ResolveResponse;
}
