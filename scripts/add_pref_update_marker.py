import os
import json
import re
import argparse
import yaml
from json_repair import repair_json
from tqdm import tqdm
from rapidfuzz import fuzz

# Add path of the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_llm import QueryLLM
import utils


def process_json_file(file_path, LLM, verbose=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_dict = data.get("Q&A", {})
    modified_qa_dict = {}  # Store only modified parts

    for conversation_key, qa_list in qa_dict.items():
        print('conversation_key', conversation_key)
        modified_qa_list = []  # Track modifications for this conversation section

        for qa in tqdm(qa_list):
            if conversation_key == 'Conversation Next Year':
                qa['More_Update'] = "No"
            else:
                reference = qa.get("Reference", {})
                # Check for keys indicating a preference fact or update.
                for ref_key in ["[Fact] Likes", "[Fact] Dislikes", "[Updated Fact] Likes", "[Updated Fact] Dislikes"]:
                    if ref_key in reference:
                        event = reference[ref_key]

                        # look for the same event mentioned in later conversation_key
                        if conversation_key == 'Init Conversation':
                            future_conversation_keys = ['Conversation Next Week', 'Conversation Next Month', 'Conversation Next Year']
                        elif conversation_key == 'Conversation Next Week':
                            future_conversation_keys = ['Conversation Next Month', 'Conversation Next Year']
                        else:
                            future_conversation_keys = ['Conversation Next Year']

                        # Check if any of the reference events appear in future conversation keys
                        qa['More_Update'] = "No"
                        for future_key in future_conversation_keys:
                            for qa_event in qa_dict[future_key]:
                                future_events = {qa_event.get("Reference", {}).get(k, "") for k in ["[Fact] Likes", "[Fact] Dislikes", "[Updated Fact] Likes", "[Updated Fact] Dislikes"]}
                                if {event} & future_events:  # Check if there's any intersection
                                    qa['More_Update'] = "Yes"
                                    if verbose:
                                        print(f"Found future update for {event} in {future_key}")
                                    break

            modified_qa_list.append(qa)

        # If modifications were made, store them
        if modified_qa_list:
            modified_qa_dict[conversation_key] = modified_qa_list

        # Step 5: Write only the modified parts back to the JSON file
        if modified_qa_dict:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                full_data = json.load(json_file)  # Load entire JSON structure
            with open(file_path, 'w', encoding='utf-8') as json_file:
                for key, new_qa_list in modified_qa_dict.items():
                    full_data["Q&A"][key] = new_qa_list  # Update only modified Q&A sections
                json.dump(full_data, json_file, indent=4)  # Write back modified JSON
                print(f"Updated file: {file_path}: {conversation_key}")


def process_all_files(directory, persona_range, LLM, verbose=False):
    if '-' in persona_range:
        min_persona, max_persona = map(int, persona_range.split('-'))
    else:
        min_persona, max_persona = int(persona_range), int(persona_range)

    for root, _, files in tqdm(os.walk(directory)):
        # if files is not a list
        if not isinstance(files, list):
            continue
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
                print(f'{utils.Colors.OKGREEN}Processing file: {os.path.join(root, file)}{utils.Colors.ENDC}')
                process_json_file(os.path.join(root, file), LLM, verbose=verbose)


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('/pool/bwjiang/memory_bench/config.yaml') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    parser = argparse.ArgumentParser(description="Validate JSON files in a directory.")
    parser.add_argument("--path", type=str, default="./data/output/", help="Directory path to search JSON files.")
    parser.add_argument("--persona_range", type=str, default=0, help="Persona ID range (e.g., 8-11).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    cmd_args = parser.parse_args()

    LLM = QueryLLM(args)
    process_all_files(cmd_args.path, cmd_args.persona_range, LLM, verbose=cmd_args.verbose)