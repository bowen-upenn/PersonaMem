#!/bin/bash

# Arrays for the parameters
idx_personas=$(seq 1 1)

# Iterate over each combination of parameters
for idx_persona in $idx_personas; do
        # Construct the command
        command="python prepare_data.py --model gpt-4o \
                 --topics therapy legalConsultation datingConsultation foodRecommendation onlineShopping studyConsultation travelPlanning bookRecommendation \
                          movieRecommendation songRecommendation sportsRecommendation homeDecoration healthConsultation \
                 --n_persona ${idx_persona} --n_samples 1 --s_persona 0 --s_samples 0"

        # Print the command for debugging/logging purposes
        echo "$command"

        # Execute the command
        eval "$command"
done