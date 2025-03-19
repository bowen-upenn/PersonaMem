import os
import json
import random
import tiktoken
import argparse
import yaml
import math
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


def reformat_coding_conversation(conversation_lines, which_format):
    """
    Special handling for conversations related to coding:
    - Group messages into full utterances.
    - Preserve code blocks as part of the Assistant's response.
    """

    utterances = []
    current_utterance = None
    in_code_block = False
    code_block_delim = "```"

    def start_new_utterance(role, content):
        return {"role": role, "content": content}

    # A helper function to finalize the current utterance.
    def flush_current_utterance():
        nonlocal current_utterance
        if current_utterance is not None:
            current_utterance["content"] = current_utterance["content"].strip()
            if current_utterance["content"]:
                utterances.append(current_utterance)
            current_utterance = None

    # Determine role from a line marker
    def get_role_from_line(line):
        if line.startswith("User:"):
            return "user"
        elif line.startswith("Assistant:"):
            return "assistant"
        elif line.startswith("[Original_Code]:"):
            return "assistant"
        return "user"

    for line in conversation_lines:
        line = line.rstrip()

        # Skip side notes.
        if line.startswith("[Side_Note]:"):
            continue

        # Check for code block start/end.
        if line.startswith(code_block_delim):
            if in_code_block:
                current_utterance["content"] += "\n" + line
                in_code_block = False
                continue
            else:
                flush_current_utterance()
                current_utterance = start_new_utterance("assistant", line)
                in_code_block = True
                continue

        if in_code_block:
            current_utterance["content"] += "\n" + line
            continue

        # Check for a new utterance marker.
        role_match = re.match(r"^(User:|Assistant:|\[Original_Code\]:)", line)
        if role_match:
            flush_current_utterance()
            role = get_role_from_line(line)
            content = re.sub(r"^(User:|Assistant:|\[Original_Code\]:)\s*", "", line)
            current_utterance = start_new_utterance(role, content)
        else:
            if current_utterance is None:
                current_utterance = start_new_utterance("user", line)
            else:
                current_utterance["content"] += "\n" + line

    flush_current_utterance()

    if which_format == 'string':
        formatted = []
        for utt in utterances:
            formatted.append(f"{utt['role'].capitalize()}: {utt['content']}")
        return "\n\n".join(formatted)
    elif which_format == 'api_dict':
        return utterances
    else:
        raise NotImplementedError(f"Unknown format: {which_format}")


def reformat_conversation(topic, conversation, which_format):
    # Ensure conversation is a list of lines
    if isinstance(conversation, str):
        conversation = conversation.splitlines()
    elif isinstance(conversation, list):
        conversation = conversation
    else:
        raise ValueError("Conversation must be a string or list of strings.")

    # Special formatting for coding topics
    if topic == 'coding':
        return reformat_coding_conversation(conversation, which_format)

    extracted_conversation = []

    if which_format == 'string':
        # Format as a pure string, removing lines that start with 'Side_Note'
        for line in conversation:
            line = line.strip()
            if not line or line.startswith("Side_Note"):
                continue
            # Remove dates in the format (dd/mm/yyyy) or dd/mm/yyyy
            line = re.sub(r'\(?\b\d{2}/\d{2}/\d{4}\b\)?', '', line).strip()
            extracted_conversation.append(line)

        return "\n".join(extracted_conversation)

    elif which_format == 'api_dict':
        # Format the list for an LLM API in a message format
        for line in conversation:
            # print(line, '\n')
            line = line.strip()
            if not line or line.startswith("Side_Note"):
                continue
            # Remove dates
            line = re.sub(r'\(?\b\d{2}/\d{2}/\d{4}\b\)?', '', line).strip()

            # Determine role based on the topic
            if topic == 'therapy':
                role = 'assistant' if line.startswith(("Therapist:", "Therapist")) else "user"
            else:
                role = 'assistant' if line.startswith(("Assistant:", "Assistant")) else "user"

            # Append to conversation list
            extracted_conversation.append({"role": role, "content": line})

        return extracted_conversation

    else:
        raise NotImplementedError(f"Unknown format: {which_format}")


def process_conversation_block(topic, conversation, which_format):
    global no_timestamp_record
    latest_timestamp = None

    # Find the latest timestamp by scanning backwards
    for line in reversed(conversation):
        if line.startswith("Side_Note") or line.startswith("[Side_Note]") or line.startswith("Side_Note:") or line.startswith("[Side_Note]:") or line.startswith("[Side_Note") or line.startswith(
                "[Side_Note:"):
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
    persona = None
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if f"persona{idx_persona}_" in file_name:
                topic = file_name.split('_')[1]
                fpath = os.path.join(root, file_name)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not persona:
                    persona = data['Expanded Persona']

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
    available_weeks = set()  # will be unlocked by chosen init blocks
    available_months = set()  # will be unlocked by chosen week blocks
    available_years = set()  # will be unlocked by chosen month blocks

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

    # Dynamic weighting function for priority
    def get_weight(block, progress_ratio):
        """ Assigns a dynamic weight based on the progress in block selection. """
        if progress_ratio < 0.25:
            return 1 if block in available_inits else 0.1
        elif progress_ratio < 0.5:
            return 1 if block in available_weeks else (0.2 if block in available_inits else 0.1)
        elif progress_ratio < 0.75:
            return 1 if block in available_months else (0.2 if block in available_weeks else 0.1)
        else:
            return 1 if block in available_years else (0.2 if block in available_months else 0.1)

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

        # Track the count of blocks per topic
        topic_counts = {}
        for block in chosen:
            fname, _ = block
            topic = fname.split('_')[1]
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Determine dynamic weights based on progress
        progress_ratio = len(chosen) / n_blocks
        weights = [get_weight(block, progress_ratio) for block in current_pool]

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Sample using weighted probabilities
        block = random.choices(current_pool, weights=normalized_weights, k=1)[0]
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
    final_blocks = [(b, candidates[b]) for b in chosen]

    # if verbose:
    #     print("Chosen conversation blocks:")
    #     for (fn, k), data in final_blocks:
    #         print(f"{fn}: {k}")
    #     print(len(final_blocks), "blocks chosen.")

    return final_blocks, persona


def topological_sort(processed_blocks, num_variants=1, verbose=False):
    def extract_topic(file_name):
        # For example, for "conversation_travelPlanning_persona0_sample0.json",
        # return "travelPlanning_persona0"
        parts = file_name.split("_")
        return "_".join(parts[1:3])

    # Map time_period to a causal order value.
    causal_order_mapping = {
        "Init Conversation": 0,
        "Conversation Next Week": 1,
        "Conversation Next Month": 2,
        "Conversation Next Year": 3,
        "Conversation": 0  # for topics like writing, email, coding
    }

    def get_causal_order(block):
        return causal_order_mapping.get(block["time_period"], float("inf"))

    # Group blocks by topic.
    topics = defaultdict(list)
    for block in processed_blocks.values():
        topic = extract_topic(block["file_name"])
        topics[topic].append(block)

    # Sort each topic's blocks in causal order.
    for topic, blocks in topics.items():
        blocks.sort(key=get_causal_order)

    variants = []
    for variant in range(num_variants):
        random.seed(variant)

        """
        Mode A refers to having long distance from the last session to its previous sessions, designed for Q&As queried within the last session
        Mode B refers to having long distance from the questions in the end to the last sessions, designed for Q&As at the END OF TEXT
        """
        if num_variants == 1:
            mode = "A"
        elif num_variants == 2:
            mode = "A" if variant == 0 else "B"
        else:
            mode = "A" if random.random() < 0.5 else "B"

        # Step 1: Randomly sample one "Conversation Next Year" session (from any topic)
        next_year_blocks = [
            block
            for blocks in topics.values()
            for block in blocks
            if block["time_period"] == "Conversation Next Year"
        ]
        if not next_year_blocks:
            raise ValueError("No 'Conversation Next Year' sessions available.")
        chosen_next_year_block = random.choice(next_year_blocks)
        chosen_topic = extract_topic(chosen_next_year_block["file_name"])

        # Step 2: For the chosen topic, arrange all its sessions in causal order.
        t_sessions = topics[chosen_topic]
        t_sessions = sorted(t_sessions, key=get_causal_order)

        # Gather sessions from other topics.
        other_sessions = defaultdict(list)
        for topic, blocks in topics.items():
            if topic != chosen_topic:
                for block in blocks:
                    other_sessions[get_causal_order(block)].append(block)
        for key in other_sessions:
            random.shuffle(other_sessions[key])
        other_sessions = [
            block for order in sorted(other_sessions.keys())
            for block in other_sessions[order]
        ]

        if mode == "A":
            # Mode A: Long distance to the previous session.
            # Step 3A: Split the chosen topic's sessions into two parts.
            # Search for the "Conversation Next Month" session.
            split_index = None
            for i, block in enumerate(t_sessions):
                if block["time_period"] == "Conversation Next Month":
                    split_index = i
                    break
            # Fallback: use the first "Conversation Next Year" if no Next Month found.
            if split_index is None:
                for i, block in enumerate(t_sessions):
                    if block["time_period"] == "Conversation Next Year":
                        split_index = i
                        break
            if split_index is None:
                split_index = len(t_sessions) - 1

            t_before = t_sessions[:split_index + 1]  # up to and including the split
            t_after = t_sessions[split_index + 1:]  # sessions after the split

            # Step 4A: Partition other sessions.
            # Allow up to 10% of t_before count to be interleaved.
            allowed_count = math.floor(0.1 * len(t_before))
            interleaved_others = other_sessions[:allowed_count]
            remaining_others = other_sessions[allowed_count:]

            # Step 5A: Interleave the allowed other sessions evenly into t_before.
            interleaved_t_before = []
            t_before_copy = t_before.copy()
            if interleaved_others and len(t_before_copy) > 1:
                gap = len(t_before_copy) / (len(interleaved_others) + 1)
                insertion_indices = [int(round(gap * (i + 1))) for i in range(len(interleaved_others))]
                idx_other = 0
                total_len = len(t_before_copy) + len(interleaved_others)
                for i in range(total_len):
                    if idx_other < len(insertion_indices) and i == insertion_indices[idx_other] + idx_other:
                        interleaved_t_before.append(interleaved_others[idx_other])
                        idx_other += 1
                    else:
                        interleaved_t_before.append(t_before_copy.pop(0))
            else:
                interleaved_t_before = t_before

            # Step 6A: Merge for Mode A.
            merged_list = interleaved_t_before + remaining_others + t_after

        else:
            # Mode B: Long distance after the last session.
            # Step 3B: Use the chosen topic's sessions as one block.
            t_topic = t_sessions.copy()  # all sessions of the chosen topic, in causal order

            # Step 4B: Allow interleaving of up to 10% of total sessions (across the entire dataset)
            # into the chosen topic block to avoid having them too contiguous.
            total_sessions_count = len(processed_blocks)
            allowed_count = math.floor(0.1 * total_sessions_count)
            interleaved_topic = []
            t_topic_copy = t_topic.copy()
            if allowed_count > 0 and len(t_topic_copy) > 1 and len(other_sessions) >= allowed_count:
                gap = len(t_topic_copy) / (allowed_count + 1)
                insertion_indices = [int(round(gap * (i + 1))) for i in range(allowed_count)]
                idx_other = 0
                total_len = len(t_topic_copy) + allowed_count
                # Use the first allowed_count sessions from other_sessions for interleaving.
                interleaved_others = other_sessions[:allowed_count]
                for i in range(total_len):
                    if idx_other < len(insertion_indices) and i == insertion_indices[idx_other] + idx_other:
                        interleaved_topic.append(interleaved_others[idx_other])
                        idx_other += 1
                    else:
                        interleaved_topic.append(t_topic_copy.pop(0))
            else:
                interleaved_topic = t_topic

            # Remove the ones used for interleaving from other_sessions.
            remaining_others = other_sessions[allowed_count:]

            # Step 5B: Merge for Mode B.
            # The chosen topic block (with slight interleaving) comes first,
            # followed by all remaining sessions from other topics.
            merged_list = interleaved_topic + remaining_others

        if verbose:
            if mode == "A":
                mode = "A - Long distance to previous session of the same topic"
            else:
                mode = "B - Long distance after the last session of this topic"
            print(f"Variant {variant + 1} (Mode {mode}):")
            sorted_info = [
                f"{block['file_name']}: {block['time_period']}"
                for block in merged_list
            ]
            print("Sorted conversation blocks:", sorted_info)
            print('-' * 50)

        variants.append(merged_list)

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


def concatenate_blocks(sorted_processed_blocks, which_format, tokenizer, all_irrelevant_contexts=None, persona=None, verbose=False):
    all_conversations = []
    num_irrelevant_tokens = 0

    for block_idx, block in enumerate(sorted_processed_blocks):
        curr_conversations = []

        # Append persona at the beginning of each block
        if persona:
            if which_format == 'string':
                curr_conversations.append(persona)
            else:
                curr_conversations.append({"role": "system", "content": "Current user persona:" + persona})

        # Insert irrelevant contexts
        # if all_irrelevant_contexts and which_format == 'api_dict':
        #     num_random_blocks = random.randint(0, 15)
        #     random_sessions = random.sample(all_irrelevant_contexts, min(num_random_blocks, len(all_irrelevant_contexts)))
        #     for session in random_sessions:
        #         key = list(session.keys())[0]  # only one key in each session
        #         if session[key]:
        #             curr_conversations.extend(session[key])
        #     # Remove all items whose content is None from curr_conversations
        #     curr_conversations = [item for item in curr_conversations if item['content'] is not None]
        #     num_irrelevant_tokens += count_tokens(" ".join([item['content'] for item in curr_conversations]), tokenizer, verbose=False)

        if which_format == 'string':
            curr_conversations.append(block["conversation"])
        else:
            curr_conversations.extend(block["conversation"])
        all_conversations.append(curr_conversations)

    # if verbose:
    #     print(f'{utils.Colors.OKGREEN}Conversations:{utils.Colors.ENDC}')
    #     print(all_conversations)
    return all_conversations, num_irrelevant_tokens


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


def compute_question_distance(sorted_processed_blocks, tokenizer, all_conversations, num_irrelevant_tokens):
    """
    We assume the questions are asked at the end of all concatenated conversation blocks.
    This function computes the distance of each question from the end to its corresponding conversation block.
    Range of distance: [0, total_blocks-1]
    """
    total_blocks = len(sorted_processed_blocks)
    flattened_all_conversations = [item for curr_conversations in all_conversations for item in curr_conversations]
    all_qa = []

    # Precompute the token counts for each conversation block and a prefix sum list.
    prefix_tokens = [0]
    for item in flattened_all_conversations:
        # Count tokens in each block's content.
        token_count = count_tokens(item['content'], tokenizer, verbose=False)
        prefix_tokens.append(prefix_tokens[-1] + token_count)

    for i, block in enumerate(sorted_processed_blocks):
        # Only keep Q&As in the last block, i.e., the current session during conversation
        # if i + 1 < total_blocks:
        #     continue

        # we assign distance to all qa in the current block
        for idx, q in enumerate(block.get('qa', [])):
            if not q:
                continue

            # For all non-last session, only allow Q&As with 'Where' == 'END OF TEXT' and no further preference updates
            if i + 1 < total_blocks:
                if ('Where' not in q) or ('Where' in q and q['Where'] != 'END OF TEXT'):
                    continue
                if ('More_Update' not in q) or ('More_Update' in q and q['More_Update'] == 'Yes'):
                    continue

            # Get where the question will be asked
            where = q['Where']

            # For all sessions except for the final one, we ignore all questions asked within the conversation
            if i + 1 < total_blocks and where != 'END OF TEXT':
                continue

            if where == 'END OF TEXT':
                block_num_q, start_index_q = total_blocks, len(flattened_all_conversations) - 1
            else:
                block_num_q, start_index_q = utils.find_string_in_list(where, flattened_all_conversations, all_conversations)

            num_tokens_q = prefix_tokens[start_index_q]
            # num_tokens_q = count_tokens(" ".join([item['content'] for item in flattened_all_conversations[:start_index_q]]), tokenizer, verbose=False)
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
                    # print('reference_utterance', reference_utterance, 'where', where, 'type', q['Type'])
                    block_num_ref, start_index_ref = utils.find_string_in_list(reference_utterance, flattened_all_conversations, all_conversations)
                    # print('start_index_ref', start_index_ref, 'len(all_conversations)', len(all_conversations))
                    # print('all_conversations[start_index_ref]', all_conversations[start_index_ref])
                    # print('reference_utterance', reference_utterance)
                else:
                    # For sequence of updates Q&A, it is a list of dictionary. We need to find the last one, i.e., the earliest one
                    all_timestamps = [key for key in q['Reference'] if key != 'full_sequence']
                    all_timestamps.sort(key=lambda x: datetime.strptime(x, "%m/%d/%Y"))
                    try:
                        reference_event = q['Reference'][all_timestamps[0]]['Conversation']
                        reference_utterance = reference_event.split('\n')[1]
                    except:
                        reference_event = q['Reference'][all_timestamps[1]]['Conversation']  # in case the earliest timestamp is not associated with a conversation
                        reference_utterance = reference_event.split('\n')[1]

                    block_num_ref, start_index_ref = utils.find_string_in_list(reference_utterance, flattened_all_conversations, all_conversations)

            num_tokens_ref = prefix_tokens[start_index_ref]
            # num_tokens_ref = count_tokens(" ".join([item['content'] for item in flattened_all_conversations[:start_index_ref]]), tokenizer, verbose=False)

            # print('len(flattened_all_conversations)', len(flattened_all_conversations), 'total_num_of_tokens', count_tokens(" ".join([item['content'] for item in flattened_all_conversations if 'content' in item]), tokenizer, verbose=False))
            # print('block_num_ref', block_num_ref, 'start_index_ref', start_index_ref, 'num_tokens_ref', num_tokens_ref)
            # print('block_num_q', block_num_q, 'start_index_q', start_index_q, 'num_tokens_q', num_tokens_q)

            num_tokens_question = count_tokens(q['Question'], tokenizer, verbose=False)
            q['distance_blocks'] = block_num_q - block_num_ref
            q['distance_tokens'] = num_tokens_q - num_tokens_ref + num_tokens_question
            q['context_length_in_tokens'] = num_tokens_q + num_tokens_question
            q['context_length_in_letters'] = len("".join([item['content'] for item in curr_context]))
            q['shared_context'] = flattened_all_conversations
            q['end_index_in_shared_context'] = start_index_q
            q['curr_context'] = curr_context
            q['num_irrelevant_tokens'] = num_irrelevant_tokens
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
        correct_answer = '(' + chr(97 + correct_index) + ')'  # + qa["Correct_Answer"] # Convert index to letter (e.g., 0 -> 'a')

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
        num_irrelevant_tokens = qa['num_irrelevant_tokens']
        where = qa['Where'] if 'Where' in qa else None
        stereotypical = qa['Stereotypical'] if 'Stereotypical' in qa else None

        yield (curr_context, question, formatted_question, correct_answer, all_options, distance_blocks, distance_tokens, question_type, topic, where, stereotypical,
               context_length_in_tokens, context_length_in_letters, shared_context, end_index_in_shared_context, num_irrelevant_tokens)