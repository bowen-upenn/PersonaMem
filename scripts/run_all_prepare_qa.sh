#!/bin/bash

# Full list of topics for reference
# bookRecommendation coding datingConsultation email familyRelations financialConsultation foodRecommendation homeDecoration \
# legalConsultation medicalConsultation movieRecommendation musicRecommendation onlineShopping sportsRecommendation \
# studyConsultation therapy travelPlanning writing \

# Full list of topics for reference
# init next_week next_month next_year all
time_period="next_month"

# Lauren
# start_persona_id=0
# end_persona_id=4  # non-inclusive

## Zoey
#start_persona_id=4
#end_persona_id=8
#
## Yuan
#start_persona_id=10
#end_persona_id=11

## Jeff
start_persona_id=14
end_persona_id=15

## Brian
#start_persona_id=16
#end_persona_id=20

# Construct the command
command="python prepare_qa.py --model gpt-4o --action qa \
         --topics sportsRecommendation \
         --n_persona ${end_persona_id} --n_samples 1 --s_persona ${start_persona_id} --s_samples 0 --time ${time_period}"

# Print the command for debugging/logging purposes
echo "$command"

# Execute the command
eval "$command"

# All errored paths:

# Error generating Q&A for reasons of change ./data/output/movieRecommendation/conversation_movieRecommendation_persona13_sample0.json:Conversation Next Week
    # Error generating Q&A for reasons of change'[Old Fact] Likes'                                                                                                                                          
    # Error generating Q&A for reasons of change'[Old Fact] Dislikes'

# Error generating Q&A for reasons of change ./data/output/movieRecommendation/conversation_movieRecommendation_persona13_sample0.json:Conversation Next Month
    # Error generating Q&A for reasons of change'[Old Fact] Likes'
    # Error generating Q&A for reasons of change'[Old Fact] Dislikes'
    # Error generating Q&A for reasons of change'[Old Fact] Likes'
    # Error generating Q&A for reasons of change'[Old Fact] Dislikes'
    # Error generating Q&A for reasons of change'[Old Fact] Dislikes'

# Error generating Q&A for reasons of change ./data/output/onlineShopping/conversation_onlineShopping_persona14_sample0.json:Conversation Next Week
    # '[Old Fact] Dislikes' 

# Error generating Q&A for reasons of change ./data/output/sportsRecommendation/conversation_sportsRecommendation_persona14_sample0.json:Conversation Next Week
    # '[Old Fact] Dislikes'   

# persoan14
#     onlineShopping
#     sportsRecommendation