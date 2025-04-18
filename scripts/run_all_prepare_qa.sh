#!/bin/bash

# Full list of topics for reference
# bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

# Full list of time_periods for reference
# init next_week next_month next_year

# Define the list of time periods
time_periods=("init" "next_week" "next_month" "next_year")

start_persona_id=0
end_persona_id=20  # non-inclusive

# Loop over each time period
for time_period in "${time_periods[@]}"; do
    # Construct the command
    command="python prepare_qa.py --model gpt-4o --action qa \
             --topics bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
                      legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
                      studyConsultation therapy travelPlanning writing \
             --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --time ${time_period} --clean"

    # Print the command for debugging/logging purposes
    echo "$command"

    # Execute the command
    eval "$command"
done