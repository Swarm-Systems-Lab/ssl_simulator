#!/usr/bin/env bash
set -euo pipefail

# Shared helpers to activate the project virtual environment.
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

activate_venv() {
	if [ ! -d .venv ]; then
		echo "Missing .venv. Run ./scripts/ci/setup-env.sh first." >&2
		exit 1
	fi
	. .venv/bin/activate
}
