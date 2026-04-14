"use client";

import React from "react";

type ButtonVariant = "primary" | "secondary";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

export function Button({ variant = "primary", className = "", ...props }: ButtonProps) {
  const base =
    "inline-flex items-center justify-center rounded-md px-3 py-2 text-sm font-medium shadow-sm " +
    "focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed";

  const styles =
    variant === "primary"
      ? "bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500"
      : "bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-400";

  return <button className={`${base} ${styles} ${className}`} {...props} />;
}
