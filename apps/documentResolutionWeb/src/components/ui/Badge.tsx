"use client";

import React from "react";

export function Badge({
  variant = "neutral",
  children,
  className = "",
}: {
  variant?: "neutral" | "accent";
  children: React.ReactNode;
  className?: string;
}) {
  const styles =
    variant === "accent"
      ? "bg-blue-50 text-blue-700 border-blue-200"
      : "bg-gray-50 text-gray-700 border-gray-200";

  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs ${styles} ${className}`}>
      {children}
    </span>
  );
}
