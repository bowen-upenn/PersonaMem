import json
import os
import re
import argparse


def clean_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Skipping invalid JSON file: {file_path}")
        return

    if "Q&A" not in data:
        print(f"Skipping file with missing 'Q&A': {file_path}")
        return

    filtered_qna = {}
    for period, entries in data["Q&A"].items():
        if isinstance(entries, list):
            filtered_qna[period] = [entry for entry in entries if entry.get("Type") not in {
                "generalizing_past_reasons_in_memory_to_new_scenarios",
                "recalling_the_reasons_behind_previous_updates"
            }]

    data["Q&A"] = filtered_qna

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Cleaned file: {file_path}")


def process_json_files(directory, persona_range):
    if '-' not in persona_range:
        min_persona = max_persona = int(persona_range) if persona_range else None
    else:
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
                clean_json(os.path.join(root, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove specific Q&A types from JSON files.")
    parser.add_argument("--path", type=str, default="./data//output/", help="Directory path to search JSON files.")
    parser.add_argument("--persona_range", type=str, default=None, help="Persona ID range (e.g., 8-11).")

    args = parser.parse_args()
    process_json_files(args.path, args.persona_range)
