#!/bin/bash

# Full list of topics for reference
contexts=("datingConsultation" "familyRelations" "financialConsultation" "foodRecommendation" "homeDecoration"
         "legalConsultation" "medicalConsultation" "movieRecommendation" "musicRecommendation" "onlineShopping" "sportsRecommendation"
         "studyConsultation" "therapy" "travelPlanning")
# contexts=("bookRecommendation")

# Lauren
# idx_personas=$(seq 1 1) # this range should be inclusive

## Zoey
#idx_personas=$(seq 4 7)
#
## Yuan
#idx_personas=$(seq 8 11)
#
## Jeff
idx_personas=$(seq 12 15)
#
## Brian
#idx_personas=$(seq 16 19)

# Iterate over each combination of parameters
for context in "${contexts[@]}"; do
    for idx_persona in $idx_personas; do
        # If the context is "writing", "coding", or "email", set time_period to "init" only
        if [[ "$context" == "writing" || "$context" == "coding" || "$context" == "email" ]]; then
            time_periods=("init")
        else
#            time_periods=("init" "next_week" "next_month" "next_year")
             time_periods=("init")
        fi

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
