export type PublicEnv = {
  NEXT_PUBLIC_API_BASE_URL: string;
  NEXT_PUBLIC_ENV: "local" | "staging" | "production";
  NEXT_PUBLIC_ENABLE_ANALYTICS: boolean;
};

function must(k: string, v: string | undefined): string {
  if (!v) throw new Error(`Missing required env var: ${k}`);
  return v;
}

export const publicEnv: PublicEnv = {
  NEXT_PUBLIC_API_BASE_URL: must("NEXT_PUBLIC_API_BASE_URL", process.env.NEXT_PUBLIC_API_BASE_URL),
  NEXT_PUBLIC_ENV: (must("NEXT_PUBLIC_ENV", process.env.NEXT_PUBLIC_ENV) as PublicEnv["NEXT_PUBLIC_ENV"]),
  NEXT_PUBLIC_ENABLE_ANALYTICS: (process.env.NEXT_PUBLIC_ENABLE_ANALYTICS ?? "false") === "true",
};
