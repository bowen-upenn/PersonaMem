#!/bin/bash

# Full list of topics for reference
# bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

start_persona_id=0
end_persona_id=20  # non-inclusive

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