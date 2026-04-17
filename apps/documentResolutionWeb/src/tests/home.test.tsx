import { describe, it, expect, vi, beforeEach } from "vitest";
import Home from "@/app/page";

const redirectMock = vi.fn();

vi.mock("next/navigation", () => ({
  redirect: (path: string) => {
    redirectMock(path);
    throw new Error("NEXT_REDIRECT");
  },
}));

describe("Home page", () => {
  beforeEach(() => {
    redirectMock.mockClear();
  });

  it("redirects to /demo", () => {
    expect(() => Home()).toThrow("NEXT_REDIRECT");
    expect(redirectMock).toHaveBeenCalledWith("/demo");
  });
});
