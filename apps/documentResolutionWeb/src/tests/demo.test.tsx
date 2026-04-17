import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import DemoPage from "@/app/demo/page";

describe("Demo page", () => {
  it("renders the acronym demo UI", () => {
    render(<DemoPage />);

    expect(
      screen.getByRole("heading", { name: /acronym resolution demo/i }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", { name: /resolve/i }),
    ).toBeInTheDocument();
  });
});
