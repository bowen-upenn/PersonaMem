import json
import os
import re
import argparse
from datetime import datetime


class Colors:
    FAIL = '\033[91m'
    END = '\033[0m'


def validate_json(file_path):
    required_subkeys = [
        "Init Conversation",
        "Conversation Next Week",
        "Conversation Next Month",
        "Conversation Next Year"
    ]
    required_types = {
        "recalling_facts_mentioned_by_the_user": 0,
        "identifying_new_things_not_mentioned_by_the_user": 0,
        # "generalizing_past_reasons_in_memory_to_new_scenarios": 0,
        # "recalling_the_reasons_behind_previous_updates": 0,
        "tracking_the_full_sequence_of_preference_updates": 0,
        "recommendation_aligned_with_users_latest_preferences": 0,
        "recalling_the_latest_user_preferences": 0
    }
    ignored_types_in_init = [
        "identifying_new_things_not_mentioned_by_the_user",
        "generalizing_past_reasons_in_memory_to_new_scenarios",
        "recalling_the_reasons_behind_previous_updates",
        "tracking_the_full_sequence_of_preference_updates",
    ]
    temp_ignored_types = [
        "generalizing_past_reasons_in_memory_to_new_scenarios",
        "recalling_the_reasons_behind_previous_updates",
    ]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"{Colors.FAIL}Invalid JSON or File Not Found: {file_path}{Colors.END}")
        return

    if "Q&A" not in data:
        print(f"{Colors.FAIL}Missing 'Q&A' key in {file_path}{Colors.END}")
        return

    for period in required_subkeys:
        if period not in data["Q&A"]:
            print(f"{Colors.FAIL}Missing key '{period}' in {file_path}{Colors.END}")
            continue

        value = data["Q&A"][period]
        if not value or len(value) == 0:
            print(f"{Colors.FAIL}Empty list under '{period}' in {file_path}{Colors.END}")
            continue

        for item in value:
            if not isinstance(item, dict) or "Topic" not in item:
                print(f"{Colors.FAIL}Invalid entry in '{period}': Each item must have 'Type' in {file_path}{Colors.END}")
                continue

            if item["Type"] not in required_types and item["Type"] not in temp_ignored_types:
                print(f"{Colors.FAIL}Invalid type in '{period}'{Colors.END}")
                continue

            if item["Type"] in required_types:
                required_types[item["Type"]] += 1

        # Check if each required_type has been mentioned
        valid = True
        for type in required_types.keys():
            if required_types[type] == 0 and not (period == "Init Conversation" and type in ignored_types_in_init):
                print(f"{Colors.FAIL}Error generating 'Type' {type} in {file_path}:{period}{Colors.END}")
                valid = False

        if valid:
            print(f"All required types are present in the file {file_path}: {period}")

        # Reset the counts for next period
        required_types = {key: 0 for key in required_types.keys()}



def process_json_files(directory="./data/output/", persona_range=None):
    min_persona, max_persona = map(int, persona_range.split('-')) if persona_range else (None, None)

    for root, _, files in os.walk(directory):
        for file in files:
            if any(excluded in file.lower() for excluded in ["writing", "email", "coding"]):
                continue

            match = re.search(r'persona(\d+)_', file)
            if match:
                persona_id = int(match.group(1))
                if min_persona is not None and max_persona is not None:
                    if not (min_persona <= persona_id <= max_persona):
                        continue

            if file.endswith(".json"):
                print(f"Validating {file}...")
                validate_json(os.path.join(root, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JSON files in a directory.")
    parser.add_argument("--path", type=str, default="./data/output/", help="Directory path to search JSON files.")
    parser.add_argument("--persona_range", type=str, default=None, help="Persona ID range (e.g., 8-11).")

    args = parser.parse_args()
    process_json_files(args.path, args.persona_range)
