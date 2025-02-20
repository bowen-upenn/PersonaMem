#!/bin/bash

# familyRelations, foodRecommendation, sportsRecommendation, studyConsultation, therapy, travelPlanning

#bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
#legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
#studyConsultation therapy travelPlanning writing \

idx_persona=4
# Construct the command
command="python prepare_data.py --model gpt-4o \
         --topics writing coding email \
         --n_persona ${idx_persona} --n_samples 1 --s_persona 3 --s_samples 0"

# Print the command for debugging/logging purposes
echo "$command"

# Execute the command
eval "$command"


#idx_personas=$(seq 0 0)
#contexts=("therapy travelPlanning")
#time_periods=("init" "next_week" "next_month" "next_year")
#
## Iterate over each combination of parameters
#for context in "${contexts[@]}"; do
#    for idx_persona in $idx_personas; do
#        for time_period in "${time_periods[@]}"; do
#            # Construct the command
#            command="python prepare_qa.py --action qa --data ${context}_persona${idx_persona}_sample0 --time ${time_period}"
#
#            # Print the command for debugging/logging purposes
#            echo "$command"
#
#            # Execute the command
#            eval "$command"
#        done
#    done
#done
