#!/usr/bin/env bash
set -euo pipefail

# Test Gitea workflows locally using act
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# Install act if not available
if ! command -v act >/dev/null 2>&1; then
	echo "Installing act..."
	curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | bash
fi

# List available workflows
WORKFLOW_DIR=".gitea/workflows"
if [ ! -d "$WORKFLOW_DIR" ]; then
	echo "No workflows directory found at $WORKFLOW_DIR"
	exit 1
fi

workflows=("$WORKFLOW_DIR"/*.yml)
if [ ${#workflows[@]} -eq 0 ]; then
	echo "No workflow files found in $WORKFLOW_DIR"
	exit 1
fi

echo "Available workflows:"
select workflow in "${workflows[@]}" "Quit"; do
	case $workflow in
		"Quit")
			echo "Exiting."
			exit 0
			;;
		*)
			if [ -n "$workflow" ]; then
				echo "Running workflow: $workflow"
				act -W "$workflow" --container-architecture linux/amd64
				break
			else
				echo "Invalid selection. Please try again."
			fi
			;;
	esac
done
