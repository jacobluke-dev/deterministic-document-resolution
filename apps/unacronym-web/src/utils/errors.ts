export type ResolveClientError = {
  status: number;
  message: string;
  details?: unknown;
};

export function isResolveClientError(e: unknown): e is ResolveClientError {
  if (typeof e !== "object" || e === null) return false;

  const status = (e as Record<string, unknown>).status;
  const message = (e as Record<string, unknown>).message;

  return typeof status === "number" && Number.isFinite(status) && typeof message === "string";
}

export type ErrorEnvelope = { message?: unknown; details?: unknown };

export function isErrorEnvelope(x: unknown): x is ErrorEnvelope {
  return typeof x === "object" && x !== null;
}
