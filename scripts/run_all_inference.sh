#!/bin/bash

models=("gpt-4o" "gpt-4o-mini")
idx_personas=$(seq 2 9)
n_blocks=$(seq 1 10)

total=$(( ${#models[@]} * 10 * 10 ))
current=0

for model in "${models[@]}"; do
    for idx_persona in $idx_personas; do
        for n_block in $n_blocks; do
            ((current++))
            echo -ne "\rProgress: $current/$total"
            python inference.py --model $model --idx_persona $idx_persona --format api_dict --n_blocks $n_block
        done
    done
done

echo -e "\nAll commands executed."
