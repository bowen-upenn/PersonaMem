#!/bin/bash

# Arrays for the parameters
#contexts=("therapy" "legal" "foodRecommendation" "datingConsultation" "travelPlanning" "onlineShopping" "studyConsultation" "writing")
contexts=("writing")
idx_personas=$(seq 0 19)
time_periods=("init" "next_week" "next_month" "next_year")

# Iterate over each combination of parameters
for context in "${contexts[@]}"; do
    for idx_persona in $idx_personas; do
        for time_period in "${time_periods[@]}"; do
            # Construct the command
            command="python prepare_qa.py --action qa --data ${context}_persona${idx_persona}_sample0 --time ${time_period}"

            # Print the command for debugging/logging purposes
            echo "$command"

            # Execute the command
            eval "$command"
        done
    done
done
