SHELL := /usr/bin/env bash

.PHONY: dev install lint typecheck test build start ci-local

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
	pnpm test

build:
	corepack enable
	pnpm build

ci-local: install lint typecheck test build
	@echo "✅ unacronym-web OK"
