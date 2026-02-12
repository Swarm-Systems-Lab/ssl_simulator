#!/usr/bin/env bash
set -euo pipefail

# Fast-fail publish using uv's built-in publishing support
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
	./scripts/ci/setup-env.sh --extras dev,release
fi

. .venv/bin/activate

if [ -z "${UV_PUBLISH_USERNAME-}" ] || [ -z "${UV_PUBLISH_PASSWORD-}" ] || [ -z "${UV_PUBLISH_REPOSITORY_URL-}" ]; then
	echo "Missing publishing credentials (UV_PUBLISH_USERNAME / UV_PUBLISH_PASSWORD / UV_PUBLISH_REPOSITORY_URL)"
	exit 1
fi

echo "Publishing artifacts from dist/ using uv"
if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
	echo "No artifacts found in dist/ - nothing to publish"
	exit 1
fi

# uv has built-in publish functionality; delegate to it and pass credentials via env
# Use uv publish — exact CLI flags are provided by uv; this will fail fast if unsupported
uv publish --repository-url "$UV_PUBLISH_REPOSITORY_URL" --username "$UV_PUBLISH_USERNAME" --password "$UV_PUBLISH_PASSWORD" dist/*

echo "Publish finished"
