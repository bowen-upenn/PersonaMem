#!/bin/bash

# Full list of topics for reference
# bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

## Lauren
# start_persona_id=1
# end_persona_id=4  # non-inclusive
# Lauren
# start_persona_id=3
# end_persona_id=4  # non-inclusive

## Zoey
#start_persona_id=4
#end_persona_id=8
#
# Yuan
#start_persona_id=8
#end_persona_id=12

## Jeff
start_persona_id=12
end_persona_id=13
#
## Brian
#start_persona_id=16
#end_persona_id=20

# Construct the command
command="python prepare_data.py --model gpt-4o \
--topics bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
studyConsultation therapy travelPlanning writing \
--n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --output_dir data/output/ "

# Print the command for debugging/logging purposes
echo "$command"

# Execute the command
eval "$command"
