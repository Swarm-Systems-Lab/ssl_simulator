#!/usr/bin/env bash
set -euo pipefail

# Development build script — assumes the shared venv is active
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
	./scripts/ci/setup-env.sh --extras dev,release
fi

. .venv/bin/activate

echo "Preparing build output directory"
rm -rf dist
mkdir -p dist

echo "Running tox build environment"
uv run tox -e build

echo "Build artifacts placed in dist/"
