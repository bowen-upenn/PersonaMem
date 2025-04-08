#!/bin/bash

# Arguments for the Python script
MODEL_NAME="gpt-4.5-preview"
BENCHMARK_SIZE="128k"
QUESTION_PATH="data/questions_${BENCHMARK_SIZE}.csv"
CONTEXT_PATH="data/shared_contexts_${BENCHMARK_SIZE}.jsonl"
RESULT_PATH="data/results/eval_results_${BENCHMARK_SIZE}_${MODEL_NAME}.csv"

# Run the Python script with the specified arguments
python "inference_standalone_openai.py" --model "$MODEL_NAME" --step "evaluate" --question_path "$QUESTION_PATH" --context_path "$CONTEXT_PATH" --result_path "$RESULT_PATH" --clean
