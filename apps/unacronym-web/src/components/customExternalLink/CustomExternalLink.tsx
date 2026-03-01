import React from "react";

type CustomExternalLinkProps = {
  href: string;
  children: React.ReactNode;
  className?: string;
  openInNewTab?: boolean;
};

export default function CustomExternalLink({
  href,
  children,
  className = "",
  openInNewTab = true,
}: CustomExternalLinkProps) {
  return (
    <a
      href={href}
      target={openInNewTab ? "_blank" : undefined}
      rel={openInNewTab ? "noopener noreferrer" : undefined}
      className={`text-blue-600 hover:underline ${className}`}
    >
      {children}
    </a>
  );
}
