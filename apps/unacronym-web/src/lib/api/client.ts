import type { ResolveRequest, ResolveResponse } from "@/lib/api/types";

export type ApiError = {
  status: number;
  message: string;
  details?: unknown;
};

export async function resolveText(
  req: ResolveRequest,
  signal?: AbortSignal,
): Promise<ResolveResponse> {
  const resp = await fetch("/api/resolve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

  if (!resp.ok) {
    const payload = await resp.json().catch(() => null);
    const message =
      payload?.message ??
      (resp.status === 401
        ? "API key required/invalid."
        : resp.status === 413
          ? "Input too large (request limit exceeded)."
          : resp.status === 422
            ? "Please paste some text."
            : "Request failed.");

    const err: ApiError = { status: resp.status, message, details: payload?.details };
    throw err;
  }

  return (await resp.json()) as ResolveResponse;
}
