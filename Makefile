SHELL := /bin/bash

# Prefer pnpm if available; fall back to npm for local dev environments.
# CI can still use pnpm via corepack/setup actions.
NODE_PKG_MGR := $(shell command -v pnpm >/dev/null 2>&1 && echo "pnpm" || echo "npm")

.PHONY: lint lint-py lint-node test test-py test-node build build-node ci

lint: lint-py lint-node

lint-py:
	python -m ruff check nanobot aether_bot_web

lint-node:
	@if [ "$(NODE_PKG_MGR)" = "pnpm" ]; then \
		pnpm -C aetherctl lint; \
	else \
		npm --prefix aetherctl run lint; \
	fi

test: test-py test-node

test-py:
	python -m pytest -q

test-node:
	@if [ "$(NODE_PKG_MGR)" = "pnpm" ]; then \
		pnpm -C aetherctl test; \
	else \
		npm --prefix aetherctl test; \
	fi

build: build-node

build-node:
	@if [ "$(NODE_PKG_MGR)" = "pnpm" ]; then \
		pnpm -C aetherctl build; \
	else \
		npm --prefix aetherctl run build; \
	fi

ci: lint test build
