"use client";

import React from "react";

export function ProgressBar({ value }: { value: number }) {
  const v = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  return (
    <div className="h-2 w-full rounded-full bg-gray-200">
      <div className="h-2 rounded-full bg-blue-600" style={{ width: `${v * 100}%` }} />
    </div>
  );
}
