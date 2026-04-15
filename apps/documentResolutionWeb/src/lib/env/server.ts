function must(k: string, v: string | undefined): string {
  if (!v) throw new Error(`Missing required env var: ${k}`);
  return v;
}

export function getServerEnv() {
  return {
    DOCUMENT_RESOLUTION_API_KEY: must("DOCUMENT_RESOLUTION_API_KEY", process.env.DOCUMENT_RESOLUTION_API_KEY),
  };
}
