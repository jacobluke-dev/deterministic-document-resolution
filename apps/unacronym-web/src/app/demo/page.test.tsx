import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import DemoPage from "@/app/demo/page";


import "@testing-library/jest-dom/vitest";

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {}, // deprecated
    removeListener: () => {}, // deprecated
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});

vi.mock("@/lib/api/client", () => ({
  resolveText: vi.fn(async () => ({
    acronyms: [
      {
        acronym: "GPU",
        first_occurrence: { start: 4, end: 7 },
        definitions: [{ text: "Graphics Processing Unit", start: 9, end: 33, confidence: 0.95, source: "extracted" }],
        occurrences: [{ start: 4, end: 7 }],
        glossary: null,
      },
    ],
    meta: { processing_ms: 1, model_version: "x", input_chars: 10 },
  })),
}));

describe("/demo", () => {
  it("resolves and renders results", async () => {
    render(<DemoPage />);

    fireEvent.change(screen.getByLabelText("Text"), { target: { value: "The GPU (Graphics Processing Unit)..." } });
    fireEvent.click(screen.getByRole("button", { name: /resolve/i }));

    // acronym appears
    expect(await screen.findByText("GPU")).toBeInTheDocument();
    expect(await screen.findByText("Graphics Processing Unit")).toBeInTheDocument();
  });
});
