import { NextResponse } from "next/server";
import { publicEnv } from "@/lib/env/public";
import {serverEnv} from "@/lib/env/server";


export async function POST(req: Request) {
  const body = await req.text();

  const upstream = new URL("/v1/resolve", publicEnv.NEXT_PUBLIC_API_BASE_URL).toString();

  const res = await fetch(upstream, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": serverEnv.UNACRONYM_API_KEY,
    },
    body,
  });

  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
  });
}
