#!/usr/bin/env bash
set -e
echo "🧹  Removing placeholder comments..."
find . -type f -name '*.py' -exec sed -i '/TODO: remove/d' {} +

echo "🗑️   Deleting empty or legacy files..."
find agents baseline_models memory metrics tools data -type f -empty -delete 2>/dev/null || true
find . -type f -regex '.*\(_old\|_backup\|milvus\).*\.py' -delete 2>/dev/null || true
find docs/notebooks -type f -name '*.ipynb' -size 0 -delete 2>/dev/null || true

echo "🧽  Clearing caches..."
rm -rf **/__pycache__ **/*.pyc **/*~ .pytest_cache 2>/dev/null || true

echo "Cleanup complete."
