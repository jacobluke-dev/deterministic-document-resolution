import type { NextConfig } from "next";

// Fail fast at build-time if required env vars are missing
import "./src/lib/env/build";

const nextConfig: NextConfig = {
  output: "standalone",
};

export default nextConfig;
