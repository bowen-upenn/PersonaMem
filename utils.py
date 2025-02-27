import torch
import numpy as np
import os
import random
import json
import re
from datetime import datetime, timedelta
from sentence_transformers import util
import ast


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def preprocess_source_data(data, topic):
    if topic == 'therapy' or topic == 'legal':
        topic_conversation = ""
        for message in data["conversation"]:
            role = message["role"]
            content = message["content"]
            topic_conversation += f"{role.capitalize()}: {content}\n\n"
    else:
        raise NotImplementedError

    return topic_conversation


def load_all_source_data(source_dir, topic):
    if topic == 'writing':
        with open(source_dir, 'r') as f:
            data = json.load(f)
        prompts = list(data.keys())  # Preload the keys
        return {'data': data, 'prompts': prompts}
    elif topic == 'coding':
        all_source_files = parse_code_files_from_txt(source_dir)
        return all_source_files
    elif topic == 'email':
        all_source_files = parse_emails_from_txt(source_dir)
        return all_source_files
    else:
        all_source_files = os.listdir(source_dir)
        return all_source_files


def load_one_source_data(source_dir, all_source_files, topic):
    # Load a random source file from the real-world data
    if topic == 'writing':
        data, prompts = all_source_files['data'], all_source_files['prompts']
        random_prompt = random.choice(prompts)
        curr_samples = data[random_prompt]
        return random.choice(curr_samples)
    elif topic == 'coding':
        random_index = random.randint(0, len(all_source_files) - 1)
        return all_source_files[random_index]['content']
    elif topic == 'email':
        random_index = random.randint(0, len(all_source_files) - 1)
        return all_source_files[random_index]
    else:
        random_idx = random.randint(0, len(all_source_files) - 1)
        selected_file = all_source_files[random_idx]
        selected_file_path = os.path.join(source_dir, selected_file)
        with open(selected_file_path, 'r', encoding='utf-8') as file:
            source_data = json.load(file)
        return source_data


def parse_code_files_from_txt(file_path):
    code_pieces = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_file = None
    content_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("File: "):
            if current_file:
                current_file["content"] = "\n".join(content_lines)
                code_pieces.append(current_file)

            file_name = line.split("File: ")[1]
            current_file = {"file_name": file_name, "line_count": 0, "content": ""}
            content_lines = []

        elif line.startswith("Line count: "):
            if current_file:
                current_file["line_count"] = int(line.split("Line count: ")[1])

        elif line.startswith("=================================================="):
            continue

        else:
            content_lines.append(line)

    if current_file:
        current_file["content"] = "\n".join(content_lines)
        code_pieces.append(current_file)

    return code_pieces


def parse_emails_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Split emails using the separator line
    emails = re.split(r'-{35,}', data)

    email_samples = []
    for email in emails:
        # Find the subject line and everything after it
        match = re.search(r'(Subject:.*)', email, re.DOTALL)
        if match:
            email_samples.append(match.group(1).strip())

    return email_samples


def process_json_from_api(response):
    # Parse JSON from API response
    response = response.strip("```json").strip("```python").strip("```").strip()

    # First, convert single-quoted keys to double-quoted keys
    # Matches patterns like: 'Key':
    # Captures the key (excluding quotes), then replaces the single quotes with double
    response = re.sub(r"'([^']+)':", r'"\1":', response)

    # Convert single-quoted values to double-quoted values
    # Using double quotes for the pattern to avoid string parsing issues:
    response = re.sub(r":\s*'([^']*)'(\s*[},])", r': "\1"\2', response)

    response = json.loads(response)
    return response


def extract_json_from_response(response, parse_json=False, parse_list=False):
    if parse_json:
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
        if json_match:
            # Extract the JSON part
            json_part = json_match.group(1).strip()
            response = process_json_from_api(json_part)
        else:
            # already in JSON format
            response = json.loads(response)
    elif parse_list:
        response = response.strip("```python").strip("```plaintext").strip()
        response = ast.literal_eval(response)
    return response


def append_json_to_file(response, output_file_path, curr_data_name, parse_json=False, parse_list=False):
    def load_existing_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return {}
        else:
            return {}

    # Load the existing JSON data from the file (if any)
    existing_json_file = load_existing_json(output_file_path)

    # if curr_data_name == 'Init Contextual Personal History':
    #     match = re.split(r'```json', response, maxsplit=1)
    #     likes_and_dislikes = match[0].strip() if match else ""
    #     existing_json_file['Likes and Dislikes'] = likes_and_dislikes

    # Extract and append the new JSON data
    parsed_response = extract_json_from_response(response, parse_json, parse_list)
    existing_json_file[curr_data_name] = parsed_response

    # Save the updated data back to the file
    with open(output_file_path, "w") as json_file:
        json.dump(existing_json_file, json_file, indent=4)


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


def extract_last_timestamp(json_response):
    json_response = json.loads(json_response)
    if isinstance(json_response, list):
        json_response = json_response[0]
    timestamps = list(json_response.keys())
    last_timestamp = max(timestamps, key=lambda x: tuple(map(int, x.split('/')[::-1])))
    return last_timestamp


def merge_timestamps(timestamps):
    if len(timestamps) == 4:
        return timestamps
    print('timestamps before merging:', timestamps)
    # Function to compare dates in MM/DD/YYYY format
    def later_date(date1, date2):
        return max(date1, date2, key=lambda x: tuple(map(int, x.split('/')[::-1])))

    assert len(timestamps) % 2 == 0
    num_conv_blocks = len(timestamps) // 2
    merged_timestamps = []
    for i in range(num_conv_blocks):
        merged_timestamps.append(later_date(timestamps[i], timestamps[i + num_conv_blocks]))

    for i, timestamp in enumerate(merged_timestamps):
        random_days = random.randint(0, 6)
        random_days = timedelta(days=random_days)
        merged_timestamps[i] = (datetime.strptime(timestamp, "%m/%d/%Y") + random_days).strftime("%m/%d/%Y")
    print('timestamps after merging:', merged_timestamps)

    return merged_timestamps


def find_most_similar_event(SentenceBERT, side_note_sentence, related_data):
    """
    The same timestamp may have multiple events, like one in the general personal history and one in the topicual one.
    This function uses SentenceBERT to locate the single event we are actually targeting.
    """
    max_similarity = -1
    most_similar_data = None

    for data in related_data:
        event_sentence = data.get("event", "")
        similarity = util.pytorch_cos_sim(
            SentenceBERT.encode(side_note_sentence, convert_to_tensor=True),
            SentenceBERT.encode(event_sentence, convert_to_tensor=True)
        )
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_data = data

    return most_similar_data


def clean_raw_writing_data(source_file, output_file):
    try:
        # Read raw data from the file
        with open(source_file, 'r') as f:
            data = f.read()

        # Process the data: Remove <newline> and backticks (`) from the content
        cleaned_data = data.replace("<newline>", "").replace("`", "").replace("''", "").replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" '", "'")

        # Save the cleaned data back to the output file
        with open(output_file, 'w') as f:
            f.write(cleaned_data)

        print(f"Data has been cleaned and saved to {output_file}.")
    except FileNotFoundError:
        print(f"Error: The file {source_file} was not found.")
    except json.JSONDecodeError:
        print("Error: The source file does not contain valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_all_writing_data(topic):
    directory_path = os.path.join("./data/output", topic)
    writing_data_files = [
        filename for filename in os.listdir(directory_path) if topic in filename
    ]
    return writing_data_files


def remove_side_notes(conversation):
    pattern = re.compile(r'^\s*["\']?\[?(?:side[ _]?notes?)\]?[^\]]*[:,\]].*$', re.IGNORECASE | re.MULTILINE)
    cleaned_conversation = [line for line in conversation if not pattern.match(line.lower())]
    return cleaned_conversation


def find_existing_persona_files(idx_persona):
    output_base_dir = "./data/output"
    topic_dirs = [
        os.path.join(output_base_dir, d)
        for d in os.listdir(output_base_dir)
        if os.path.isdir(os.path.join(output_base_dir, d))
    ]

    matching_file = None
    selected_data = None

    # Loop over each topic directory and each file inside it
    for topic_dir in topic_dirs:
        for file_name in os.listdir(topic_dir):
            if f"_persona{idx_persona}_" in file_name:
                file_path = os.path.join(topic_dir, file_name)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                if "General Personal History Next Year" in data:
                    matching_file = file_path
                    selected_data = data
                    break  # Stop searching this directory if we found a match
        if matching_file:
            break  # Stop searching further directories

    if matching_file:
        print(f'Loaded persona file from {matching_file}')

        persona = selected_data.get("Original Persona")
        expanded_persona = selected_data.get("Expanded Persona")

        # Retrieve the first timestamp from "General Persona History Next Year" if available
        if "Init General Personal History" in selected_data:
            start_time = next(iter(selected_data["Init General Personal History"].keys()))
        else:
            start_time = None

        print(f'Found an existing persona file for persona {idx_persona}.')
        return {
            'persona': persona,
            'expanded_persona': expanded_persona,
            'start_time': start_time,
            'init_general_personal_history': selected_data.get("Init General Personal History"),
            'general_personal_history_next_week': selected_data.get("General Personal History Next Week"),
            'general_personal_history_next_month': selected_data.get("General Personal History Next Month"),
            'general_personal_history_next_year': selected_data.get("General Personal History Next Year"),
        }
    else:
        print(f"No persona file with 'General Personal History Next Year' found for persona {idx_persona}.")
        return None


def get_all_context_names():
    base_path = './data/output'
    sub_folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    return sub_folders


def get_all_file_names(base_folder, context=""):
    file_names = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if context != "":
                if context in file:
                    file_names.append(os.path.join(root, file))
            else:
                file_names.append(os.path.join(root, file))
    return file_names


def clean_up_subdirectories():
    base_path = './data/output'

    # Traverse through subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):  # Check for JSON files
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Remove the file
                print(f"Removed: {file_path}")


def clean_up_one_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed: {file_path}")
    else:
        print(f"File not found: {file_path}")


def find_string_in_list(data, target, sorted_processed_blocks=None):
    # Check if the data is a list of dictionaries
    if isinstance(data, list) and isinstance(data[0], dict):
        for index, item in enumerate(data):
            if item.get('content') == target:
                return 0, index
    # if isinstance(data, list) and isinstance(data[0], dict):
    #     # print('sorted_processed_blocks', sorted_processed_blocks[0]['conversation'])
    #     for index, item in enumerate(data):
    #         if item.get('content') == target:
    #             # print('index', index, len(sorted_processed_blocks), sum(len(block['conversation']) for block in sorted_processed_blocks))
    #             # # Find block index
    #             # for block_index, block in enumerate(sorted_processed_blocks):
    #             #     for entry in block['conversation']:
    #             #         if target in entry['content']:
    #             block_index = 0
    #             return block_index, index

    # Check if the data is a list of strings
    elif isinstance(data, list) and isinstance(data[0], str):
        for block_num, data_block in enumerate(data):
            start_index = data_block.find(target)
            if start_index != -1:
                return block_num, start_index
            else:
                continue

    print('target not found')
    return -1  # Return -1 if not found