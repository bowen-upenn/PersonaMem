import os
import json
import random
import tiktoken
import argparse
import yaml
import re
import torch
from datetime import datetime, timedelta
from collections import defaultdict

import utils

"""
This file contains helper functions needed to concatenate multiple blocks of various topics 
of the same persona before the inference step, as well as some testing codes.
"""

# Global variable to keep track of the number of conversation blocks without timestamps (writing topics)
no_timestamp_record = 0


def parse_date(date_str):
    return datetime.strptime(date_str, "%m/%d/%Y").date()


def reformat_conversation(topic, conversation, which_format):
    if which_format == 'string':
        # Format as a pure string, removing lines that start with 'Side_Note'
        extracted_conversation = "\n".join([line for line in conversation if not line.startswith("Side_Note")])
    elif which_format == 'api_dict':
        # Format the list for an LLM API in a message format
        extracted_conversation = []
        for line in conversation:
            if not line.startswith("Side_Note"):
                if topic == 'therapy':
                    role = 'assistant' if line.startswith("Therapist") or line.startswith("Therapist:") else "user"
                    extracted_conversation.append({"role": role, "content": line})
                else:
                    role = 'assistant' if line.startswith("Assistant") or line.startswith("Assistant:") else "user" # in writing topics, all original samples are also included in user
                    extracted_conversation.append({"role": role, "content": line})
    else:
        raise NotImplementedError("Unknown format: {}".format(which_format))

    return extracted_conversation


def process_conversation_block(topic, conversation, which_format):
    global no_timestamp_record
    latest_timestamp = None

    # Find the latest timestamp by scanning backwards
    for line in reversed(conversation):
        if line.startswith("Side_Note") or line.startswith("[Side_Note]") or line.startswith("Side_Note:") or line.startswith("[Side_Note]:") or line.startswith("[Side_Note") or line.startswith("[Side_Note:"):
            match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if match:
                latest_timestamp = match.group(0)  # Extract the matched date
                break
            else:
                latest_timestamp = None  # No date found

    if latest_timestamp is None:
        if topic not in ['writing', 'email', 'coding']:
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
        reformatted_conversation.append(reformat_conversation(topic, cleaned_conversation, format))

    return reformatted_conversation, latest_timestamp


def load_n_conversation_blocks(idx_persona, n_blocks, base_dir="./data/output", verbose=False):
    # Load all candidates
    candidates = {}
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if f"persona{idx_persona}_" in file_name:
                topic = file_name.split('_')[1]
                fpath = os.path.join(root, file_name)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if topic == 'writing' or topic == "email" or topic == "coding":
                    # For writing, we have only "Conversation"
                    if "Conversation" in data:
                        candidates[(file_name, "Conversation")] = data["Conversation"]
                else:
                    # Regular topics
                    if "Init Conversation" in data:
                        candidates[(file_name, "Init Conversation")] = data["Init Conversation"]
                    if "Conversation Next Week" in data:
                        candidates[(file_name, "Conversation Next Week")] = data["Conversation Next Week"]
                    if "Conversation Next Month" in data:
                        candidates[(file_name, "Conversation Next Month")] = data["Conversation Next Month"]
                    if "Conversation Next Year" in data:
                        candidates[(file_name, "Conversation Next Year")] = data["Conversation Next Year"]

    if len(candidates) == 0:
        raise ValueError("No conversation blocks found for the given persona.")

    # Separate by category
    init_candidates = {}
    week_candidates = {}
    month_candidates = {}
    year_candidates = {}

    # For writing topic, treat "Conversation" as the init-level block
    for (fname, key), val in candidates.items():
        topic = fname.split('_')[1]
        if topic == "writing" or topic == "email" or topic == "coding":
            # Only one level: treat as init equivalent
            if key == "Conversation":
                init_candidates[(fname, key)] = val
        else:
            if key == "Init Conversation":
                init_candidates[(fname, key)] = val
            elif key == "Conversation Next Week":
                week_candidates[(fname, key)] = val
            elif key == "Conversation Next Month":
                month_candidates[(fname, key)] = val
            elif key == "Conversation Next Year":
                year_candidates[(fname, key)] = val

    # chosen will store the selected blocks
    chosen = set()

    # We'll keep track of available blocks in each tier
    available_inits = set(init_candidates.keys())  # start with all init-level blocks
    available_weeks = set()   # will be unlocked by chosen init blocks
    available_months = set()  # will be unlocked by chosen week blocks
    available_years = set()   # will be unlocked by chosen month blocks

    # Helper functions to unlock next-level blocks
    def unlock_week_blocks(fname):
        # If this init block's next-week block exists, add it
        wk_key = (fname, "Conversation Next Week")
        # For writing topic, there's no next-week block.
        if wk_key in week_candidates:
            available_weeks.add(wk_key)

    def unlock_month_blocks(fname):
        mn_key = (fname, "Conversation Next Month")
        if mn_key in month_candidates:
            available_months.add(mn_key)

    def unlock_year_blocks(fname):
        yr_key = (fname, "Conversation Next Year")
        if yr_key in year_candidates:
            available_years.add(yr_key)

    # Since we must always start from init conversation, the first chosen block must be from init.
    # The loop will continue until we have n_blocks chosen.
    while len(chosen) < n_blocks:
        # Determine the current pool of available blocks:
        # According to the user's suggestion, at any point, we can pick:
        # - Any remaining init blocks
        # - Any week blocks that are unlocked by chosen init blocks
        # - Any month blocks unlocked by chosen week blocks
        # - Any year blocks unlocked by chosen month blocks
        current_pool = list(available_inits | available_weeks | available_months | available_years)

        if len(current_pool) == 0:
            # No more blocks to choose from and we haven't reached n_blocks
            raise ValueError("Cannot reach n_blocks, ran out of available candidates.")

        block = random.choice(current_pool)
        if block in chosen:
            # If somehow block is already chosen (should not happen if we handle sets correctly), skip
            continue

        # Choose this block
        chosen.add(block)

        # Remove from whichever set it belongs to
        if block in available_inits:
            available_inits.remove(block)
            # If it's init-level (or writing-level), unlock next-week block for its topic
            fname = block[0]
            # writing topic won't have next-week, but calling unlock won't hurt
            unlock_week_blocks(fname)

        elif block in available_weeks:
            available_weeks.remove(block)
            # Unlock month block for its topic
            fname = block[0]
            unlock_month_blocks(fname)

        elif block in available_months:
            available_months.remove(block)
            # Unlock year block for its topic
            fname = block[0]
            unlock_year_blocks(fname)

        elif block in available_years:
            available_years.remove(block)
            # Year block is the last in chain, no further unlock

    # Once we have chosen n_blocks, we can return them
    final_blocks = [ (b, candidates[b]) for b in chosen ]

    # if verbose:
        # print("Chosen conversation blocks:")
        # for (fn, k), data in final_blocks:
        #     print(f"{fn}: {k}")

    return final_blocks


def topological_sort(processed_blocks, num_variants=10, verbose=False):
    def extract_topic(file_name):
        return "_".join(file_name.split("_")[1:3])  # Extracts the topic, e.g., "homeDecoration_persona0"

    def causal_order(blocks):
        order = {"Init Conversation": 0, "Conversation Next Week": 1, "Conversation Next Month": 2, "Conversation Next Year": 3}
        return sorted(blocks, key=lambda b: order.get(b["time_period"], float("inf")))

    variants = []

    for _ in range(num_variants):
        all_blocks = list(processed_blocks.values())  # Copy of blocks

        # Shuffle blocks randomly before grouping them by topic
        random.shuffle(all_blocks)

        topic_groups = defaultdict(list)
        for block in all_blocks:
            topic = extract_topic(block["file_name"])
            topic_groups[topic].append(block)

        sorted_blocks = []
        for topic, blocks in topic_groups.items():
            random.shuffle(blocks)  # Shuffle topic blocks before ordering
            sorted_topic_blocks = causal_order(blocks)
            sorted_blocks.extend(sorted_topic_blocks)

        random.shuffle(sorted_blocks)  # Ensure different valid topological orders across variants

        if verbose:
            print(f'Variant {_ + 1}:')
            print(f'Sorted conversation blocks: {[block["file_name"] + ": " + block["time_period"] for block in sorted_blocks]}')
            print('-' * 50)

        variants.append(list(sorted_blocks))

    return variants


def get_order_mapping(original_blocks, sorted_blocks):
    """
    Generate a mapping from the original order to the sorted order, using both file_name and time_period.

    :param original_blocks: List of blocks in the original order.
    :param sorted_blocks: List of blocks in the sorted order.
    :return: Dictionary mapping original indices to sorted indices.
    """
    original_indices = {(block["file_name"], block["time_period"]): i for i, block in enumerate(original_blocks)}
    sorted_indices = {(block["file_name"], block["time_period"]): i for i, block in enumerate(sorted_blocks)}

    mapping = {original_indices[key]: sorted_indices[key] for key in original_indices}
    return mapping


def concatenate_blocks(sorted_processed_blocks, which_format, all_irrelevant_contexts=None, verbose=False):
    all_conversations = []
    for block_idx, block in enumerate(sorted_processed_blocks):
        curr_conversations = []

        # Insert irrelevant contexts
        if all_irrelevant_contexts and which_format == 'api_dict':
            num_random_blocks = random.randint(0, 20)
            random_sessions = random.sample(all_irrelevant_contexts, min(num_random_blocks, len(all_irrelevant_contexts)))
            for session in random_sessions:
                key = list(session.keys())[0]   # only one key in each session
                if session[key]:
                    curr_conversations.extend(session[key])

        if which_format == 'string':
            curr_conversations.append(block["conversation"])
        else:
            curr_conversations.extend(block["conversation"])
        all_conversations.append(curr_conversations)

    # if verbose:
    #     print(f'{utils.Colors.OKGREEN}Conversations:{utils.Colors.ENDC}')
    #     print(all_conversations)
    return all_conversations


def count_tokens(all_strings, tokenizer, verbose=False):
    # all_strings = "\n\n".join(all_strings)
    tokens = tokenizer.encode(all_strings)
    if verbose:
        print(f"{utils.Colors.OKGREEN}Number of tokens: {len(tokens)} on gpt-4o tokenizer{utils.Colors.ENDC}")
    return len(tokens)
    
    
def extract_qa(base_dir, topic, file_name, time_period):
    with open(os.path.join(base_dir, os.path.join(topic, file_name)), "r", encoding="utf-8") as f:
        data = json.load(f)

    qa = data['Q&A'][time_period]
    return qa


def compute_question_distance(sorted_processed_blocks, tokenizer, all_conversations):
    """
    We assume the questions are asked at the end of all concatenated conversation blocks.
    This function computes the distance of each question from the end to its corresponding conversation block.
    Range of distance: [0, total_blocks-1]
    """
    total_blocks = len(sorted_processed_blocks)
    flattened_all_conversations = [item for curr_conversations in all_conversations for item in curr_conversations]
    all_qa = []

    for i, block in enumerate(sorted_processed_blocks):
        # Only keep Q&As in the last block, i.e., the current session during conversation
        if i + 1 < total_blocks:
            continue

        # we assign distance to all qa in the current block
        for idx, q in enumerate(block.get('qa', [])):
            if not q:
                continue

            # Get where the question will be asked
            where = q['Where']
            if where == 'END OF TEXT':
                block_num_q, start_index_q = i, len(flattened_all_conversations) - 1
            else:
                print('\n\nDuring text')
                block_num_q, start_index_q = utils.find_string_in_list(where, flattened_all_conversations, all_conversations)

            num_tokens_q = count_tokens(" ".join([item['content'] for item in flattened_all_conversations[:start_index_q]]), tokenizer, verbose=False)
            curr_context = flattened_all_conversations[:start_index_q]

            # Get where the reference information will be
            if 'Reference' not in q:
                continue
            else:
                if block['topic'] == 'writing' or block['topic'] == 'email' or block['topic'] == 'coding':
                    reference_utterance = sorted_processed_blocks[i]['conversation'][0]['content']
                    block_num_ref, start_index_ref = utils.find_string_in_list(reference_utterance, flattened_all_conversations, all_conversations)
                elif 'Conversation' in q['Reference']:
                    reference_event = q['Reference']['Conversation']
                    reference_utterance = reference_event.split('\n')[1]
                    block_num_ref, start_index_ref = utils.find_string_in_list(reference_utterance, flattened_all_conversations, all_conversations)
                    # print('all_conversations[start_index_ref]', all_conversations[start_index_ref])
                    # print('reference_utterance', reference_utterance)
                else:
                    # For sequence of updates Q&A, it is a list of dictionary. We need to find the last one, i.e., the earliest one
                    all_timestamps = [key for key in q['Reference'] if key != 'full_sequence']
                    all_timestamps.sort(key=lambda x: datetime.strptime(x, "%m/%d/%Y"))
                    reference_event = q['Reference'][all_timestamps[0]]['Conversation']
                    # print('reference_event', reference_event)
                    reference_utterance = reference_event.split('\n')[1]
                    # print('reference_utterance', reference_utterance)
                    block_num_ref, start_index_ref = utils.find_string_in_list(reference_utterance, flattened_all_conversations, all_conversations)

            num_tokens_ref = count_tokens(" ".join([item['content'] for item in flattened_all_conversations[:start_index_ref]]), tokenizer, verbose=False)

            # print('len(flattened_all_conversations)', len(flattened_all_conversations), 'total_num_of_tokens', count_tokens(" ".join([item['content'] for item in flattened_all_conversations if 'content' in item]), tokenizer, verbose=False))
            # print('block_num_ref', block_num_ref, 'start_index_ref', start_index_ref, 'num_tokens_ref', num_tokens_ref,)
            # print('block_num_q', block_num_q, 'start_index_q', start_index_q, 'num_tokens_q', num_tokens_q)

            q['distance_blocks'] = block_num_q - block_num_ref
            q['distance_tokens'] = num_tokens_q - num_tokens_ref + count_tokens(q['Question'], tokenizer, verbose=False)
            q['context_length_in_tokens'] = num_tokens_q + count_tokens(q['Question'], tokenizer, verbose=False)
            q['context_length_in_letters'] = len(" ".join([item['content'] for item in curr_context]))
            q['shared_context'] = flattened_all_conversations
            q['end_index_in_shared_context'] = start_index_q
            q['curr_context'] = curr_context
            # print('len(curr_context)', len(curr_context), "context_length", q['context_length'])
            all_qa.append(q)

    return all_qa, flattened_all_conversations


def question_loader(qa_list):
    """
    Generator function that acts as a data loader and yields one formatted Q&A string at a time.
    Args:
        qa_list (list): A list of dictionaries containing the Q&A data from extract_qa.
    Yields:
        str: A string with the question and all candidate answers in a multiple-choice format.
    """
    for qa in qa_list:
        # For a group of static_factual type of questions, it is accompanied by a shared reference, which is not a question
        if 'Type' not in qa:
            continue
        # Skip generative questions, which is not in the multiple-choice format
        if qa['Type'] == 'new_content_generative':
            continue

        # Select three incorrect answers randomly if there are more than three
        incorrect_answers = random.sample(qa["Incorrect_Answers"], min(3, len(qa["Incorrect_Answers"])))

        # Combine correct answer with incorrect answers
        options = [qa["Correct_Answer"]] + incorrect_answers
        random.shuffle(options)

        # Find the correct answer's option
        correct_index = options.index(qa["Correct_Answer"])
        correct_answer = '(' + chr(97 + correct_index) + ')' #+ qa["Correct_Answer"] # Convert index to letter (e.g., 0 -> 'a')

        # Create the multiple-choice question string
        question = qa["Question"]
        formatted_question = f"Question: {question}\nAnswer:\n" + "\n".join(
            [f"({chr(97 + i)}) {option}" for i, option in enumerate(options)]
        )
        formatted_question += "\n.Respond with the correct option, including both the letter (a), (b), (c), or (d). Do not include other information."
        all_options = [f"({chr(97 + i)}) {option}" for i, option in enumerate(options)]

        distance_blocks = qa['distance_blocks']
        distance_tokens = qa['distance_tokens']
        question_type = qa['Type']
        topic = qa['Topic']
        context_length_in_tokens = qa['context_length_in_tokens']
        context_length_in_letters = qa['context_length_in_letters']
        shared_context = qa['shared_context']
        end_index_in_shared_context = qa['end_index_in_shared_context']
        curr_context = qa['curr_context']
        where = qa['Where'] if 'Where' in qa else None

        yield curr_context, question, formatted_question, correct_answer, all_options, distance_blocks, distance_tokens, question_type, topic, where, context_length_in_tokens, context_length_in_letters, shared_context, end_index_in_shared_context
