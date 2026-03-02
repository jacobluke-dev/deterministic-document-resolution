"use client";

import React from "react";

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border bg-white p-4 ${className}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium text-gray-900">{title}</div>
          {subtitle ? <div className="text-xs text-gray-600">{subtitle}</div> : null}
        </div>
        {right ? <div className="shrink-0">{right}</div> : null}
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}
