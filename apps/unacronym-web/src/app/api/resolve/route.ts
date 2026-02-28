import { publicEnv } from "@/lib/env/public";
import { getServerEnv } from "@/lib/env/server";

export async function POST(req: Request) {
  const { UNACRONYM_API_KEY } = getServerEnv();

  const body = await req.text();
  const upstream = new URL("/v1/resolve", publicEnv.NEXT_PUBLIC_API_BASE_URL).toString();

  const res = await fetch(upstream, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": UNACRONYM_API_KEY,
    },
    body,
  });

  return new Response(await res.text(), {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
  });
}
