FROM node:20-alpine AS deps
WORKDIR /repo

# workspace-level files
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./

# app manifest
COPY apps/unacronym-web/package.json apps/unacronym-web/package.json

RUN corepack enable && pnpm -C apps/unacronym-web install --frozen-lockfile

FROM node:20-alpine AS builder
WORKDIR /repo
COPY --from=deps /repo/node_modules ./node_modules
COPY --from=deps /repo/apps/unacronym-web/node_modules ./apps/unacronym-web/node_modules

# copy only the web app source (keeps build context smaller/cleaner)
COPY apps/unacronym-web ./apps/unacronym-web

# build-time env
ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_ENV
ARG NEXT_PUBLIC_ENABLE_ANALYTICS

ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_ENV=$NEXT_PUBLIC_ENV
ENV NEXT_PUBLIC_ENABLE_ANALYTICS=$NEXT_PUBLIC_ENABLE_ANALYTICS

RUN corepack enable && pnpm -C apps/unacronym-web build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /repo/apps/unacronym-web/.next/standalone ./
COPY --from=builder /repo/apps/unacronym-web/.next/static ./.next/static
COPY --from=builder /repo/apps/unacronym-web/public ./public

EXPOSE 3000

CMD ["node", "apps/unacronym-web/server.js"]
