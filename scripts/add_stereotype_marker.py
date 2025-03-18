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

    # Step 2: Extract persona and preferences.
    persona = data.get("Expanded Persona", {})
    if not isinstance(persona, str):
        persona = str(persona)

    # Extract contextual personal history preferences.
    context_keys = [
        "Init Contextual Personal History",
        "Contextual Personal History Next Week",
        "Contextual Personal History Next Month",
        "Contextual Personal History Next Year"
    ]

    preference_set = set()
    for key in context_keys:
        context_dict = data.get(key, {})
        if isinstance(context_dict, dict):
            for inner_key, inner_val in context_dict.items():
                if isinstance(inner_val, dict):
                    for fact_key in ["[Fact] Likes", "[Fact] Dislikes", "[Updated Fact] Likes", "Updated [Fact] Dislikes"]:
                        if fact_key in inner_val:
                            value = inner_val[fact_key].strip()
                            preference_set.add('Likes ' + value)
                            preference_set.add('Dislikes ' + value)
                            break

    # New list of preferences is the set of values extracted from the contextual history.
    preference_set = str(preference_set)
    if verbose:
        print('persona', persona)
        print('\n\npreference_set', preference_set)

    # Step 3: Prepare data for LLM query.
    query_data = {
        'persona': persona,
        'preferences': preference_set,
    }

    # Query the LLM.
    response = LLM.query_llm(step='find_stereotype', data=query_data, verbose=verbose)

    # Clean and process the LLM response.
    match = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    response = match.group(1) if match else response
    response = response.strip().replace('\n', '')
    if '=' in response:
        response = re.sub(r'^\s*\w+\s*=\s*', '', response, count=1).strip()

    # Repair JSON if necessary and load the Python list.
    response = repair_json(response)
    try:
        stereotypical_list = json.loads(response)
        print("Extracted stereotypical list:", stereotypical_list)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        stereotypical_list = []

    # Step 4: Process the Q&A section.
    qa_dict = data.get("Q&A", {})
    modified_qa_dict = {}  # Store only modified parts

    for conversation_key, qa_list in qa_dict.items():
        modified_qa_list = []  # Track modifications for this conversation section

        for qa in tqdm(qa_list):
            reference = qa.get("Reference", {})
            # Check for keys indicating a preference fact or update.
            for ref_key in ["[Fact] Likes", "[Fact] Dislikes", "[Updated Fact] Likes", "[Updated Fact] Dislikes"]:
                if ref_key in reference:
                    pref_value = reference[ref_key]
                    # print('reference', reference)

                    is_stereotypical = False
                    if '[stereotypical]' in pref_value.lower():
                        is_stereotypical = True
                        print(f"Preference '{pref_value}' is stereotypical: {is_stereotypical}")
                    else:
                        # Check if the preference is in the stereotypical list with a matching label.
                        for item in stereotypical_list:
                            if isinstance(item, dict):
                                stereotype_pref = item.get('preference', '').strip().lower()
                                # if stereotype_pref begins with Likes or Dislikes, remove the first word and the space
                                if stereotype_pref.startswith('likes '):
                                    stereotype_pref = stereotype_pref[6:]
                                elif stereotype_pref.startswith('dislikes '):
                                    stereotype_pref = stereotype_pref[9:]

                                # Apply fuzzy matching: Accept match if similarity is above 90%
                                similarity_score = fuzz.ratio(pref_value, stereotype_pref)

                                if similarity_score >= 90:  # High-confidence match
                                    # print('item', item)
                                    if (item.get('label').lower().strip() == "likes" and ref_key in ["[Fact] Likes", "[Updated Fact] Likes"]
                                            or item.get('label').lower().strip() == "dislikes" and ref_key in ["[Fact] Dislikes", "[Updated Fact] Dislikes"]):
                                        is_stereotypical = True
                                        print(f"Preference '{pref_value}' is stereotypical: {is_stereotypical}")
                                        break

                    # Add the Stereotypical field accordingly
                    qa['Stereotypical'] = "Yes" if is_stereotypical else "No"
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
