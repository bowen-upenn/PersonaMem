import torch
import numpy as np
import os
import random
import json
import re
from datetime import datetime, timedelta


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def preprocess_source_data(data, context):
    if context == 'therapy' or context == 'legal':
        context_conversation = ""
        for message in data["conversation"]:
            role = message["role"]
            content = message["content"]
            context_conversation += f"{role.capitalize()}: {content}\n\n"
    else:
        raise NotImplementedError

    return context_conversation


def load_source_data(source_dir):
    # Load a random source file from the real-world data
    all_source_files = os.listdir(source_dir)
    random_idx = random.randint(0, len(all_source_files) - 1)
    selected_file = all_source_files[random_idx]
    selected_file_path = os.path.join(source_dir, selected_file)
    with open(selected_file_path, 'r', encoding='utf-8') as file:
        source_data = json.load(file)
    return source_data


def append_json_to_file(response, output_file_path, curr_data_name, parse_json=False):
    def load_existing_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return {}
        else:
            return {}


    def extract_json_from_response(response, curr_data_name, existing_json_file, parse_json=False):
        if parse_json:
            # Use regex to extract the JSON part enclosed by "```json" and "```"
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)

            if json_match:
                # Extract the JSON part
                json_part = json_match.group(1).strip()

                try:
                    # Parse the extracted JSON string into a Python dictionary
                    parsed_json = json.loads(json_part)
                    # Add the parsed JSON content to the existing data under the user-specified key
                    if curr_data_name == 'Expand History and Conversation':
                        key = 'Expanded General Personal History' if 'Expanded General Personal History' in parsed_json else 'Expanded_General_Personal_History'
                        existing_json_file['Expanded General Personal History'] = parsed_json[key]
                        key = 'Expanded Contextual Personal History' if 'Expanded Contextual Personal History' in parsed_json else 'Expanded_Contextual_Personal_History'
                        existing_json_file['Expanded Contextual Personal History'] = parsed_json[key]
                        key = 'Expanded Conversation' if 'Expanded Conversation' in parsed_json else 'Expanded_Conversation'
                        existing_json_file['Expanded Conversation'] = parsed_json[key]
                    else:
                        existing_json_file[curr_data_name] = parsed_json

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {curr_data_name}: {e}")
            else:
                print(f"No JSON content found for {curr_data_name}")
        else:
            # Add the entire response as a string
            existing_json_file[curr_data_name] = response
        return existing_json_file


    # Load the existing JSON data from the file (if any)
    existing_json_file = load_existing_json(output_file_path)

    # Extract and append the new JSON data
    appended_json_file = extract_json_from_response(response, curr_data_name, existing_json_file, parse_json)

    # Save the updated data back to the file
    with open(output_file_path, "w") as json_file:
        json.dump(appended_json_file, json_file, indent=4)


def pick_a_random_time():
    # Skewed random selection towards recent years
    weights = np.array([i for i in range(1, 2011-1920+1)])
    weights[-20:] *= 3
    weights = weights / weights.sum()
    year = random.choices(
        population=range(1920, 2011),
        weights=weights,
        k=1
    )[0]

    # Random month and day
    month = random.randint(1, 12)
    day = random.randint(1, 28 if month == 2 else 30 if month in [4, 6, 9, 11] else 31)

    return f"{month:02d}/{day:02d}/{year}"


def pick_a_random_time_within_a_year(input_date):
    # Convert input string to datetime object
    input_date = datetime.strptime(input_date, "%m/%d/%Y")

    # Generate a random timedelta within a year (365 days in both directions)
    days_difference = random.randint(0, 365)
    new_date = input_date + timedelta(days=days_difference)

    # Return the new date in the same format
    return new_date.strftime("%m/%d/%Y")