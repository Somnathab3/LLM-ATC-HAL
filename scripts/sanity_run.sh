#!/usr/bin/env bash
set -e
echo "🔄  Running unit tests..."
pytest -q

echo "🚀  Running demo CLI..."
llm-atc demo

echo "📊  Running tiny shift benchmark..."
llm-atc shift-benchmark --config experiments/shift_experiment_config.yaml --tiers in_distribution --n 3

echo "✅  Sanity run completed without errors."
