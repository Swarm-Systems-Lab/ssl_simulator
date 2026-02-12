#!/usr/bin/env bash
set -euo pipefail

# CI publish script — use twine with credentials from environment
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
	./scripts/ci/setup-env.sh --extras dev,release
fi

. .venv/bin/activate

if [ -z "${TWINE_USERNAME-}" ] || [ -z "${TWINE_PASSWORD-}" ] || [ -z "${TWINE_REPOSITORY_URL-}" ]; then
	echo "Missing publishing credentials (TWINE_USERNAME / TWINE_PASSWORD / TWINE_REPOSITORY_URL)"
	exit 1
fi

if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
	echo "No artifacts found in dist/ - nothing to publish"
	exit 1
fi

echo "Publishing artifacts from dist/ using twine"
python -m twine upload --non-interactive --repository-url "$TWINE_REPOSITORY_URL" dist/*

echo "Publish finished"
