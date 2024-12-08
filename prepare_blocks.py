import os
import json
import random
import tiktoken
import argparse
import yaml
import re
from datetime import datetime

import utils

# Global variable to keep track of the number of conversation blocks without timestamps (writing context)
no_timestamp_record = 0


def parse_date(date_str):
    return datetime.strptime(date_str, "%m/%d/%Y").date()


def reformat_conversation(context, conversation, which_format):
    if which_format == 'string':
        # Format as a pure string, removing lines that start with 'Side_Note'
        extracted_conversation = "\n".join([line for line in conversation if not line.startswith("Side_Note")])
    elif which_format == 'api_dict':
        # Format the list for an LLM API in a message format
        extracted_conversation = []
        for line in conversation:
            if not line.startswith("Side_Note"):
                if context == 'therapy':
                    role = "assistant" if line.startswith("Lawyer Assistant") else "user"
                    extracted_conversation.append({"role": role, "content": line})
                elif context == 'legal':
                    role = "assistant" if line.startswith("Lawyer") else "user"
                    extracted_conversation.append({"role": role, "content": line})
                elif context == 'writing':
                    role = "assistant" if line.startswith("Assistant") else "user" # user includes both 'User' and '[Original_Sentence]'
                    extracted_conversation.append({"role": role, "content": line})
                else:
                    raise NotImplementedError("Unknown context: {}".format(context))
    else:
        raise NotImplementedError("Unknown format: {}".format(which_format))

    return extracted_conversation


def process_conversation_block(context, conversation, which_format):
    global no_timestamp_record
    latest_timestamp = None

    # Find the latest timestamp by scanning backwards
    for line in reversed(conversation):
        if line.startswith("Side_Note") or line.startswith("[Side_Note]"):
            match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if match:
                latest_timestamp = match.group(0)  # Extract the matched date
            else:
                latest_timestamp = None  # No date found
            break

    if latest_timestamp is None:
        if context != 'writing':
            raise ValueError("No Side_Note timestamp found in conversation block.")
        else:
            formatted_last_four = f"{no_timestamp_record:04d}"
            latest_timestamp = '00-00-' + formatted_last_four
            no_timestamp_record += 1

    cleaned_conversation = []
    for line in conversation:
        if line.startswith("Side_Note") or line.startswith("[Side_Note]"):
            # Extract the timestamp from this side note line
            parts = line.split()
            timestamp = parts[-1]

            # Append date_str to the previous line in processed_conversation
            if cleaned_conversation:
                cleaned_conversation[-1] = cleaned_conversation[-1] + f" ({timestamp})"
            # remove the side notes from the conversation
        else:
            cleaned_conversation.append(line)

    which_format = ['string'] if which_format == 'string' else ['api_dict', 'string']
    reformatted_conversation = []
    for format in which_format:
        reformatted_conversation.append(reformat_conversation(context, cleaned_conversation, format))

    return reformatted_conversation, latest_timestamp


def load_n_conversation_blocks(idx_persona, n_blocks, base_dir="./data/output", verbose=False):
    candidates = {}

    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            context = file_name.split('_')[1]
            if f"persona{idx_persona}_" in file_name:
                fpath = os.path.join(root, file_name)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if context == 'writing':
                    all_keys = ["Conversation"]
                else:
                    all_keys = ["Init Conversation", "Conversation Next Week", "Conversation Next Month", "Conversation Next Year"]
                for key in all_keys:
                    if key in data:
                        candidates[(file_name, key)] = data[key]

    if n_blocks > len(candidates):
        raise ValueError("Not enough conversation blocks available.")

    # Randomly sample the desired number of blocks
    chosen_blocks = random.sample(list(candidates.items()), n_blocks)
    if verbose:
        print(f'{utils.Colors.OKGREEN}Chosen conversation blocks:{utils.Colors.ENDC}')
        print([f"{block[0][0]}: {block[0][1]}" for block in chosen_blocks])
    return chosen_blocks


def topological_sort(processed_blocks, verbose=False):
    # The writing context does not have timestamps, so we need to handle it separately
    writing_blocks, other_blocks = [], []
    for latest_ts, block in processed_blocks.items():
        if latest_ts[:2] == '00':
            writing_blocks.append(block)
        else:
            other_blocks.append(block)

    sorted_processed_blocks = sorted(other_blocks, key=lambda x: parse_date(x['last_timestamp']))

    # Randomly insert writing blocks into the sorted list
    for writing_block in writing_blocks:
        random_index = random.randint(0, len(sorted_processed_blocks))
        sorted_processed_blocks.insert(random_index, writing_block)

    if verbose:
        print(f'{utils.Colors.OKGREEN}Sorted last time stamps:{utils.Colors.ENDC}')
        print([block['last_timestamp'] for block in sorted_processed_blocks])
        print(f'{utils.Colors.OKGREEN}Sorted conversation blocks:{utils.Colors.ENDC}')
        print([f"{block['file_name']}: {block['time_period']}" for block in sorted_processed_blocks])
    return sorted_processed_blocks


def concatenate_blocks(sorted_processed_blocks, which_format, verbose=False):
    if which_format == 'string':
        all_conversations = "\n\n".join([block["conversation"] for block in sorted_processed_blocks])
    else:
        all_conversations = []
        for block in sorted_processed_blocks:
            all_conversations.extend(block["conversation"])

    if verbose:
        print(f'{utils.Colors.OKGREEN}Conversations:{utils.Colors.ENDC}')
        print(all_conversations)
    return all_conversations


def count_tokens(all_strings):
    all_strings = "\n\n".join(all_strings)
    tokens = tokenizer.encode(all_strings)
    print(f"{utils.Colors.OKGREEN}Number of tokens: {len(tokens)} on {args['models']['llm_model']} tokenizer{utils.Colors.ENDC}")


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--idx_persona', type=int, default=0, help='Index of the persona')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of conversation blocks')
    parser.add_argument('--format', type=str, default='string', help='Output conversation format: string or api_dict')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()
    idx_persona = cmd_args.idx_persona
    n_blocks = cmd_args.n_blocks
    which_format = cmd_args.format
    verbose = cmd_args.verbose

    tokenizer = tiktoken.encoding_for_model(args['models']['llm_model'])

    # Gather all candidate conversation blocks
    base_dir = "./data/output"
    chosen_blocks = load_n_conversation_blocks(idx_persona, n_blocks, base_dir, verbose)

    # Process each chosen conversation block
    processed_blocks_dict = {}
    all_strings = []
    for (file_name, time_period), conversation in chosen_blocks:
        context = file_name.split('_')[1]
        processed_conversation, latest_ts = process_conversation_block(context, conversation, which_format)
        processed_blocks_dict[latest_ts] = {
            "conversation": processed_conversation[0],
            "file_name": file_name,
            "time_period": time_period,
            "last_timestamp": latest_ts,
            "context": context
        }
        all_strings.append(processed_conversation[-1])

    # Topological sort chosen conversation blocks by the latest timestamp
    sorted_processed_blocks = topological_sort(processed_blocks_dict, verbose)

    # Concatenate all conversation blocks
    all_conversations = concatenate_blocks(sorted_processed_blocks, which_format, verbose)
    count_tokens(all_strings)

