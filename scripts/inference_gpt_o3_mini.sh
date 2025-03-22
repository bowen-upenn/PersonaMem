#!/bin/bash

# Arguments for the Python script
MODEL_NAME="o3-mini"
QUESTION_PATH="data/questions_128k.csv"
CONTEXT_PATH="data/shared_contexts_128k.jsonl"
RESULT_PATH="data/results/eval_results_128k_${MODEL_NAME}.csv"

# Run the Python script with the specified arguments
python "inference.py" --model "$MODEL_NAME" --step "evaluate" --question_path "$QUESTION_PATH" --context_path "$CONTEXT_PATH" --result_path "$RESULT_PATH" --clean
