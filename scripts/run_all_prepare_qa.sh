#!/bin/bash

# Arrays for the parameters

#bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
#legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
#studyConsultation therapy travelPlanning writing \
contexts=("email" "coding" "writing")

idx_personas=$(seq 0 0)
time_periods=("init")

# Iterate over each combination of parameters
for context in "${contexts[@]}"; do
    for idx_persona in $idx_personas; do
        for time_period in "${time_periods[@]}"; do
            # Construct the command
            command="python prepare_qa.py --action qa --data ${context}_persona${idx_persona}_sample0 --time ${time_period} --verbose"

            # Print the command for debugging/logging purposes
            echo "$command"

            # Execute the command
            eval "$command"
        done
    done
done
