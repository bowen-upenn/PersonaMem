#!/bin/bash

# Function to clean up a single file
clean_up_one_file() {
  local file_path=$1
  if [ -f "$file_path" ]; then
    rm "$file_path"
    echo "Removed: $file_path"
  else
    echo "File not found: $file_path"
  fi
}

# TODO: Enter the failed data files here
file_list=(
  "data/output/foodRecommendation/conversation_foodRecommendation_persona8_sample0.json"
)

# Clean up all failed files in the list
for file in "${file_list[@]}"; do
  clean_up_one_file "$file"
done

# Get the total number of files
total_files=${#file_list[@]}
current=0

# Loop through each file in the list
for file in "${file_list[@]}"; do
  # Increment the counter
  current=$((current + 1))

  # Show progress
  progress=$((current * 100 / total_files))
  echo -ne "Processing file $current/$total_files ($progress%)\r"

  # Extract the context from the filename
  context=$(echo "$file" | grep -oP '(?<=output/)[^/]+')

  # Extract the persona_id from the filename
  persona_id=$(echo "$file" | grep -oP 'persona\K\d+')

  # Increment the persona_id for n_persona
  n_persona=$((persona_id + 1))

  # Execute the Python command
  python prepare_data.py --context "$context" --n_persona "$n_persona" --s_persona "$persona_id"

done
echo "Done"