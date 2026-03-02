import { describe, it, expect, vi } from "vitest";
import { resolveText } from "@/lib/api/client";

describe("resolveText", () => {
  it("happy path returns data", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ acronyms: [], meta: { processing_ms: 1, model_version: "x", input_chars: 1 } }), { status: 200 })
    ) as any);

    const res = await resolveText({ text: "GPU" });
    expect(res.meta.processing_ms).toBe(1);
  });

  it("401 bubbles message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ message: "API key required/invalid." }), { status: 401, headers: { "content-type": "application/json" } })
    ) as any);

    await expect(resolveText({ text: "GPU" })).rejects.toMatchObject({ status: 401 });
  });

  it("413 bubbles message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ message: "Input too large" }), { status: 413, headers: { "content-type": "application/json" } })
    ) as any);

    await expect(resolveText({ text: "x" })).rejects.toMatchObject({ status: 413 });
  });

  it("422 bubbles message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ message: "Please paste some text." }), { status: 422, headers: { "content-type": "application/json" } })
    ) as any);

    await expect(resolveText({ text: "" })).rejects.toMatchObject({ status: 422 });
  });
});
