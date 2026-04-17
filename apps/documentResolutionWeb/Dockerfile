FROM node:20-alpine AS deps
WORKDIR /repo

# workspace-level files
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./

# app manifest
COPY apps/documentResolutionWeb/package.json apps/documentResolutionWeb/package.json

RUN corepack enable && pnpm -C apps/documentResolutionWeb install --frozen-lockfile

FROM node:20-alpine AS builder
WORKDIR /repo
COPY --from=deps /repo/node_modules ./node_modules
COPY --from=deps /repo/apps/documentResolutionWeb/node_modules ./apps/documentResolutionWeb/node_modules

# copy only the web app source (keeps build context smaller/cleaner)
COPY apps/documentResolutionWeb ./apps/documentResolutionWeb

# build-time env
ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_ENV
ARG NEXT_PUBLIC_ENABLE_ANALYTICS

ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_ENV=$NEXT_PUBLIC_ENV
ENV NEXT_PUBLIC_ENABLE_ANALYTICS=$NEXT_PUBLIC_ENABLE_ANALYTICS

RUN corepack enable && pnpm -C apps/documentResolutionWeb build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=local

COPY --from=builder /repo/apps/documentResolutionWeb/.next/standalone ./
COPY --from=builder /repo/apps/documentResolutionWeb/.next/static ./apps/documentResolutionWeb/.next/static
COPY --from=builder /repo/apps/documentResolutionWeb/public ./apps/documentResolutionWeb/public

EXPOSE 3001

CMD ["node", "apps/documentResolutionWeb/server.js"]
