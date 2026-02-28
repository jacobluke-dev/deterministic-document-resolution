SHELL := /usr/bin/env bash

.PHONY: dev install lint typecheck test build start ci-local

NEXT_PUBLIC_API_BASE_URL ?= http://api:8000
NEXT_PUBLIC_ENV ?= local
NEXT_PUBLIC_ENABLE_ANALYTICS ?= false

dev:
	pnpm dev

start:
	pnpm start

install:
	corepack enable
	pnpm install --frozen-lockfile

lint:
	corepack enable
	pnpm lint

typecheck:
	corepack enable
	pnpm exec tsc -p tsconfig.json --noEmit

test:
	corepack enable
	pnpm test -- --reporter=verbose

build:
	corepack enable
	NEXT_PUBLIC_API_BASE_URL="$(NEXT_PUBLIC_API_BASE_URL)" \
	NEXT_PUBLIC_ENV="$(NEXT_PUBLIC_ENV)" \
	NEXT_PUBLIC_ENABLE_ANALYTICS="$(NEXT_PUBLIC_ENABLE_ANALYTICS)" \
	pnpm build

ci-local: install lint typecheck test build
	@echo "✅ unacronym-web OK"
