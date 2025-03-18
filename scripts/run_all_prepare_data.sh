#!/bin/bash

# Full list of topics for reference
# bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

# Lauren
# start_persona_id=2
# end_persona_id=3  # non-inclusive

## Zoey
start_persona_id=4
end_persona_id=5
#
## Yuan
#start_persona_id=10
#end_persona_id=11

## Jeff
#start_persona_id=12
#end_persona_id=16

## Brian
#start_persona_id=16
#end_persona_id=17

# Construct the command
command="python prepare_data.py --model gpt-4o \
         --topics travelPlanning \
         --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --output_dir data/output/ "

# Print the command for debugging/logging purposes
echo "$command"

# Execute the command
eval "$command"
