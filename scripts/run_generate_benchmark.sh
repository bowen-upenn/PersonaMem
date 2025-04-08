#!/bin/bash

# Limit memory usage to 32 GB (in KB)
ulimit -v $((32 * 1024 * 1024))

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [medium|large]"
    exit 1
fi

# Set n_blocks based on the user input
if [ "$1" == "medium" ]; then
    n_blocks=20
elif [ "$1" == "large" ]; then
    n_blocks=60
elif [ "$1" == "small" ]; then
    n_blocks=10
else
    echo "Invalid argument. Please specify 'medium' or 'large'."
    exit 1
fi

start_persona_id=0
end_persona_id=20  # non-inclusive

# Loop through idx_persona from 0 to 19
for ((idx_persona=start_persona_id; idx_persona<end_persona_id; idx_persona++)); do
    if [ "$idx_persona" -eq 0 ]; then
        echo "Saving benchmark data for idx_persona=$idx_persona with n_blocks=$n_blocks from scratch"
        python inference.py --step prepare --model gpt-4o-mini --idx_persona "$idx_persona" --n_blocks "$n_blocks" --n_variants 2 --filter_questions --clean --verbose
    else
        echo "Saving benchmark data for idx_persona=$idx_persona with n_blocks=$n_blocks"
        python inference.py --step prepare --model gpt-4o-mini --idx_persona "$idx_persona" --n_blocks "$n_blocks" --n_variants 2 --filter_questions --verbose  # Never add --clean here
    fi
done

# Usage
# bash scripts/run_all_inference.sh medium to generate the medium size benchmark up to 128k context window
# bash scripts/run_all_inference.sh large to generate the large size benchmark up to 1M context window