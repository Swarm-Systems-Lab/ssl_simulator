#!/usr/bin/env bash
# validate_docs.sh - Validate the documentation build output without building.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SITE_DIR="${SITE_DIR:-$PROJECT_ROOT/site}"
MKDOCS_CONFIG="${MKDOCS_CONFIG:-$PROJECT_ROOT/mkdocs.yml}"

echo "Validating documentation build output..."

cd "$PROJECT_ROOT"

if [[ ! -f "$MKDOCS_CONFIG" ]]; then
  echo "Error: mkdocs config not found at $MKDOCS_CONFIG" >&2
  exit 1
fi

if [[ ! -d "$SITE_DIR" ]]; then
  echo "Error: site directory not found at $SITE_DIR" >&2
  echo "Build the docs first (e.g., mkdocs build)." >&2
  exit 1
fi

if [[ ! -s "$SITE_DIR/index.html" ]]; then
  echo "Error: missing or empty index.html in $SITE_DIR" >&2
  exit 1
fi

html_count=$(find "$SITE_DIR" -type f -name '*.html' | wc -l | tr -d ' ')
if [[ "$html_count" -lt 2 ]]; then
  echo "Error: expected more than one HTML file in $SITE_DIR" >&2
  exit 1
fi

if [[ -f "$SITE_DIR/search/search_index.json" ]]; then
  if [[ ! -s "$SITE_DIR/search/search_index.json" ]]; then
    echo "Warning: search index exists but is empty" >&2
  fi
else
  echo "Warning: search index not found at $SITE_DIR/search/search_index.json" >&2
fi

echo "Documentation validation successful."
echo "Validated $html_count HTML files in $SITE_DIR"
