import json
import re
import sys
import os


def filter_valid_dates(data):
    """
    Filters JSON-like data:
    - If it's a list, keep only its dictionary elements.
    - If it's a dictionary, remove keys that are not in MM/DD/YYYY format.
    """
    if isinstance(data, str):
        data = json.loads(data)

    # If data is a list, filter dictionaries and pick the one with the most keys
    if isinstance(data, list):
        dict_list = [item for item in data if isinstance(item, dict)]
        if not dict_list:
            return {}  # Return empty dict if no valid dict is found
        data = max(dict_list, key=lambda d: len(d.keys()), default={})

    # If data is a dict, remove invalid keys
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if re.match(r'\d{2}/\d{2}/\d{4}', k)}

    return data

def list_files_in_directory(root_path):
    target_files = []
    for foldername, subfolders, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.json'):
                target_files.append(os.path.join(foldername, filename))
    return target_files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        
        # loop through all folder
        target_files = list_files_in_directory(root_path)
        for file_path in target_files:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
            try:
                # Extract "Init General Personal History" key
                input_data = parsed_data.get("Init General Personal History", {})

                filtered_data = filter_valid_dates(input_data)

                # Replace old data with the filtered one
                parsed_data["Init General Personal History"] = filtered_data

                # Save back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, indent=2)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error: {e}")
    else:
        print("Usage: python scripts/fix_json_syntax_in_personal_hist.py <file_path>")