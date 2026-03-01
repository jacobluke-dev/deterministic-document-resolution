"use client";

import React from "react";

export function Modal({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <button
        className="absolute inset-0 bg-black/40"
        aria-label="Close modal"
        onClick={onClose}
      />
      <div className="relative z-10 w-full max-w-2xl rounded-lg bg-white p-4 shadow-lg">
        <div className="flex items-start justify-between gap-3">
          <div className="text-sm font-semibold text-gray-900">{title}</div>
          <button
            onClick={onClose}
            className="rounded px-2 py-1 text-sm text-gray-700 hover:bg-gray-100"
          >
            Close
          </button>
        </div>
        <div className="mt-3">{children}</div>
      </div>
    </div>
  );
}
