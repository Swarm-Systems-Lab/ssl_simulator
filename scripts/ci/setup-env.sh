#!/usr/bin/env bash
set -euo pipefail

# Shared environment bootstrap using uv for venv management.
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA_LIST=("dev" "lint" "tests" "type-checking" "pre-commit")
ALL_EXTRAS=0
FROZEN=1
EXTRA_PACKAGES=()

usage() {
	echo "Usage: $0 [--extras <comma-separated>] [--all-extras] [--no-frozen] [--with <pkg> ...]" >&2
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--groups)
			# Backward compatibility shim for old flag name
			[[ $# -ge 2 ]] || usage
			IFS=',' read -r -a EXTRA_LIST <<< "$2"
			shift 2
			;;
		--extras)
			[[ $# -ge 2 ]] || usage
			IFS=',' read -r -a EXTRA_LIST <<< "$2"
			shift 2
			;;
		--all-groups)
			ALL_EXTRAS=1
			shift
			;;
		--all-extras)
			ALL_EXTRAS=1
			shift
			;;
		--no-frozen)
			FROZEN=0
			shift
			;;
		--with)
			[[ $# -ge 2 ]] || usage
			EXTRA_PACKAGES+=("$2")
			shift 2
			;;
		*)
			usage
			;;
	esac
done

# Install uv if not available
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

# Install just if not available
if ! command -v just >/dev/null 2>&1; then
	curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
	export PATH="$HOME/.local/bin:$PATH"
fi

# Create uv-managed venv if it doesn't exist
if [ ! -d .venv ]; then
	uv venv .venv
fi

# Activate and sync dependencies
. .venv/bin/activate

SYNC_ARGS=(sync)
((FROZEN)) && SYNC_ARGS+=("--frozen")

if ((ALL_EXTRAS)); then
	SYNC_ARGS+=("--all-extras")
elif [ ${#EXTRA_LIST[@]} -gt 0 ]; then
	for extra in "${EXTRA_LIST[@]}"; do
		SYNC_ARGS+=("--extra" "$extra")
	done
fi

uv "${SYNC_ARGS[@]}"

if [ ${#EXTRA_PACKAGES[@]} -gt 0 ]; then
	uv add "${EXTRA_PACKAGES[@]}"
fi
