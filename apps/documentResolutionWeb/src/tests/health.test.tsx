import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import HealthPage from "../app/health/page";

describe("Health page", () => {
  it("renders", () => {
    render(<HealthPage />);
    expect(screen.getByRole("heading", { name: /health/i })).toBeInTheDocument();
  });
});
