#!/bin/bash

# Full list of topics for reference
# bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

# Full list of topics for reference
# init next_week next_month next_year all
time_period="all"

## Lauren
#start_persona_id=2
#end_persona_id=3  # non-inclusive

## Zoey
start_persona_id=5
end_persona_id=6
#
# Yuan
# start_persona_id=8
# end_persona_id=12

## Jeff
#start_persona_id=12
#end_persona_id=16

## Brian
#start_persona_id=16
#end_persona_id=20

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
