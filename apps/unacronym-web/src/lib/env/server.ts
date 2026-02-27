function must(k: string, v: string | undefined): string {
  if (!v) throw new Error(`Missing required env var: ${k}`);
  return v;
}

export const serverEnv = {
  UNACRONYM_API_KEY: must("UNACRONYM_API_KEY", process.env.UNACRONYM_API_KEY),
};
