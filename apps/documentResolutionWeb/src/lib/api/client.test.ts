import { describe, it, expect, vi, afterEach } from "vitest";
import { resolveText } from "@/lib/api/client";

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubFetch(fn: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
  vi.stubGlobal("fetch", fn as typeof fetch);
}

describe("resolveText", () => {
  it("happy path returns data", async () => {
    stubFetch(async () =>
      new Response(
        JSON.stringify({
          acronyms: [],
          meta: { processing_ms: 1, model_version: "x", input_chars: 1 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const res = await resolveText({ text: "GPU" });
    expect(res.meta.processing_ms).toBe(1);
  });

  it("401 bubbles message", async () => {
    stubFetch(async () =>
      new Response(
        JSON.stringify({ message: "API key required/invalid." }),
        { status: 401, headers: { "content-type": "application/json" } },
      ),
    );

    await expect(resolveText({ text: "GPU" })).rejects.toMatchObject({ status: 401 });
  });

  it("413 bubbles message", async () => {
    stubFetch(async () =>
      new Response(
        JSON.stringify({ message: "Input too large" }),
        { status: 413, headers: { "content-type": "application/json" } },
      ),
    );

    await expect(resolveText({ text: "x" })).rejects.toMatchObject({ status: 413 });
  });

  it("422 bubbles message", async () => {
    stubFetch(async () =>
      new Response(
        JSON.stringify({ message: "Please paste some text." }),
        { status: 422, headers: { "content-type": "application/json" } },
      ),
    );

    await expect(resolveText({ text: "" })).rejects.toMatchObject({ status: 422 });
  });
});
