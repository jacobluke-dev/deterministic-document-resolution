import type { NextConfig } from "next";
import path from "path";
import "./src/lib/env/build";

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname, "../.."),
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
