#!/bin/bash

# TODO: Enter the failed data files here
file_list=(
  "./data/output/legal/conversation_legal_persona18_sample0"
  "./data/output/legal/conversation_legal_persona10_sample0.json"
  "./data/output/legal/conversation_legal_persona11_sample0.json"
  "./data/output/legal/conversation_legal_persona4_sample0.json"
  "./data/output/legal/conversation_legal_persona0_sample0.json"
  "./data/output/legal/conversation_legal_persona2_sample0.json"
  "./data/output/legal/conversation_legal_persona16_sample0.json"
  "./data/output/foodRecommendation/conversation_foodRecommendation_persona15_sample0.json"
  "./data/output/foodRecommendation/conversation_foodRecommendation_persona8_sample0.json"
  "./data/output/foodRecommendation/conversation_foodRecommendation_persona18_sample0.json"
  "./data/output/foodRecommendation/conversation_foodRecommendation_persona6_sample0.json"
  "./data/output/foodRecommendation/conversation_foodRecommendation_persona19_sample0.json"
  "./data/output/foodRecommendation/conversation_foodRecommendation_persona2_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona10_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona14_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona16_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona19_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona2_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona1_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona4_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona5_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona7_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona6_sample0.json"
  "./data/output/datingConsultation/conversation_datingConsultation_persona9_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona3_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona1_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona0_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona11_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona8_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona16_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona15_sample0.json"
  "./data/output/travelPlanning/conversation_travelPlanning_persona5_sample0.json"
  "./data/output/onlineShopping/conversation_onlineShopping_persona2_sample0.json"
  "./data/output/onlineShopping/conversation_onlineShopping_persona1_sample0.json"
  "./data/output/onlineShopping/conversation_onlineShopping_persona10_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona8_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona6_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona4_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona2_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona10_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona14_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona17_sample0.json"
  "./data/output/studyConsultation/conversation_studyConsultation_persona19_sample0.json"
)

# Total number of files
total_files=${#file_list[@]}
current=0
declare -A processed_data_names

# Loop through each file in the list
for file in "${file_list[@]}"; do
  # Increment the counter
  current=$((current + 1))

  # Show progress
  progress=$((current * 100 / total_files))
  echo -ne "Processing file $current/$total_files ($progress%)\r"

  # Extract the data name from the filename
  # For example:
  #   conversation_foodRecommendation_persona8_sample0.json
  # becomes
  #   foodRecommendation_persona8_sample0
  data_name=$(basename "$file" .json | sed 's/^conversation_//')

  # If we've already processed this data_name, skip it
  if [[ -n "${processed_data_names[$data_name]}" ]]; then
    continue
  fi

  # Mark this data_name as processed
  processed_data_names[$data_name]=1

  # For each of the specified times, run the prepare_qa.py command
  for time_option in init next_week next_month next_year; do
    python prepare_qa.py --action qa --data "$data_name" --time "$time_option"
  done

done

echo -e "\nDone"
