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

# def topological_sort(processed_blocks, new_content_samples, num_variants=3, verbose=False):
#     def random_date(start, end):
#         delta = end - start
#         random_days = random.randint(0, delta.days)
#         return start + timedelta(days=random_days)
#
#     all_timestamps = []
#     for latest_ts, block in processed_blocks.items():
#         if latest_ts[:2] != '00':  # Only process valid timestamps
#             all_timestamps.append(parse_date(block['last_timestamp']))
#     all_timestamps.sort()
#
#     if not all_timestamps:
#         raise ValueError("No valid timestamps found. Ensure some blocks have timestamps.")
#
#     earliest_timestamp = all_timestamps[0]
#     latest_timestamp = all_timestamps[-1]
#
#     variants = []
#
#     for _ in range(num_variants):
#         all_blocks = list(processed_blocks.values())  # Copy of blocks
#         # all_strings_copy = list(all_strings)  # Ensure strings match the blocks
#         new_content_samples_copy = list(new_content_samples)  # Ensure alignment
#
#         # Shuffle blocks before assigning timestamps to add randomness
#         shuffled_data = list(zip(all_blocks, new_content_samples_copy))
#         random.shuffle(shuffled_data)  # Shuffle before assigning timestamps
#         all_blocks, new_content_samples_copy = zip(*shuffled_data)
#
#         all_blocks = list(all_blocks)
#         # all_strings_copy = list(all_strings_copy)
#         new_content_samples_copy = list(new_content_samples_copy)
#
#         for block in all_blocks:
#             if block['last_timestamp'][:2] == '00':  # Assign random timestamp if missing
#                 random_timestamp = random_date(earliest_timestamp, latest_timestamp)
#                 block['last_timestamp'] = random_timestamp.strftime("%m/%d/%Y")
#
#         all_timestamps = [parse_date(block['last_timestamp']) for block in all_blocks]
#
#         # Shuffle blocks with the same timestamp to introduce more variation
#         combined = list(zip(all_timestamps, all_blocks, new_content_samples_copy))
#         combined.sort(key=lambda x: x[0])  # Sort primarily by timestamp
#
#         # If multiple items have the same timestamp, shuffle them within that timestamp
#         grouped_by_timestamp = {}
#         for item in combined:
#             ts = item[0]
#             if ts not in grouped_by_timestamp:
#                 grouped_by_timestamp[ts] = []
#             grouped_by_timestamp[ts].append(item)
#
#         sorted_blocks = []
#         for ts in sorted(grouped_by_timestamp.keys()):  # Iterate through sorted timestamps
#             random.shuffle(grouped_by_timestamp[ts])  # Shuffle within timestamp group
#             sorted_blocks.extend(grouped_by_timestamp[ts])
#
#         # Unpack sorted components while maintaining order
#         all_timestamps_sorted, all_blocks_sorted, new_content_samples_sorted = zip(*sorted_blocks)
#
#         if verbose:
#             print(f'Variant {_ + 1}:')
#             print(f'Sorted timestamps: {[block["last_timestamp"] for block in all_blocks_sorted]}')
#             print(f'Sorted conversation blocks: {[block["file_name"] + ": " + block["time_period"] for block in all_blocks_sorted]}')
#             print('-' * 50)
#
#         variants.append((list(all_blocks_sorted), list(new_content_samples_sorted)))
#
#     return variants


def concatenate_blocks(sorted_processed_blocks, which_format, all_irrelevant_contexts=None, verbose=False):
    all_conversations = []
    for block_idx, block in enumerate(sorted_processed_blocks):
        # Insert irrelevant contexts
        if all_irrelevant_contexts and which_format == 'api_dict':
            num_random_blocks = random.randint(0, 20)
            random_sessions = random.sample(all_irrelevant_contexts, min(num_random_blocks, len(all_irrelevant_contexts)))
            for session in random_sessions:
                key = list(session.keys())[0]   # only one key in each session
                if session[key]:
                    all_conversations.extend(session[key])

        if which_format == 'string':
            all_conversations.append(block["conversation"])
        else:
            all_conversations.extend(block["conversation"])

        # If this block is about writing new samples, we also append the new sample into the whole topic
        # if block['topic'] == 'writing' or block['topic'] == 'email' or block['topic'] == 'coding':
        #     print('After sort new_content_samples: ', block_idx, sorted_new_content_samples)
        #     assert sorted_new_content_samples[block_idx]   # should not be empty
        #     original_sample = sorted_new_content_samples[block_idx]["Original Sample"]
        #     updated_sample = sorted_new_content_samples[block_idx]["Updated Sample"]
        #     if which_format == 'string':
        #         all_conversations.append("Help summarize our conversation by showing the original sample and the updated sample here again."
        #                                  "\n\nOriginal Sample" + original_sample + "\n\nUpdated Sample" + updated_sample + "\n\n")
        #     else:
        #         all_conversations.append({"role": "user", "content": "Help summarize our conversation by showing the original sample and the updated sample here again."})
        #         all_conversations.append({"role": "assistant", "content": "Original Sample:\n\n" + original_sample + "\n\n" + "Updated Sample:\n\n" + updated_sample + "\n\n"})

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
                block_num_q, start_index_q = i, len(all_conversations) - 1
            else:
                block_num_q, start_index_q = utils.find_string_in_list(all_conversations, where, sorted_processed_blocks)

            num_tokens_q = count_tokens(" ".join([item['content'] for item in all_conversations[:start_index_q]]), tokenizer, verbose=False)
            curr_context = all_conversations[:start_index_q]

            # Get where the reference information will be
            if 'Reference' not in q:
                continue
            else:
                if block['topic'] == 'writing' or block['topic'] == 'email' or block['topic'] == 'coding':
                    reference_utterance = sorted_processed_blocks[i]['conversation'][0]['content']
                    block_num_ref, start_index_ref = utils.find_string_in_list(all_conversations, reference_utterance, sorted_processed_blocks)
                elif 'Conversation' in q['Reference']:
                    reference_event = q['Reference']['Conversation']
                    reference_utterance = reference_event.split('\n')[1]
                    block_num_ref, start_index_ref = utils.find_string_in_list(all_conversations, reference_utterance, sorted_processed_blocks)
                else:
                    # For sequence of updates Q&A, it is a list of dictionary. We need to find the last one, i.e., the earliest one
                    all_timestamps = [key for key in q['Reference'] if key != 'full_sequence']
                    all_timestamps.sort(key=lambda x: datetime.strptime(x, "%m/%d/%Y"))
                    reference_event = q['Reference'][all_timestamps[0]]['Conversation']
                    # print('reference_event', reference_event)
                    reference_utterance = reference_event.split('\n')[1]
                    # print('reference_utterance', reference_utterance)
                    block_num_ref, start_index_ref = utils.find_string_in_list(all_conversations, reference_utterance, sorted_processed_blocks)

            num_tokens_ref = count_tokens(" ".join([item['content'] for item in all_conversations[:start_index_ref]]), tokenizer, verbose=False)

            q['distance_blocks'] = block_num_q - block_num_ref
            q['distance_tokens'] = num_tokens_q - num_tokens_ref + count_tokens(q['Question'], tokenizer, verbose=False)
            q['context_length'] = num_tokens_q + count_tokens(q['Question'], tokenizer, verbose=False)
            q['curr_context'] = curr_context
            # print('len(curr_context)', len(curr_context), "context_length", q['context_length'])
            all_qa.append(q)

    return all_qa

# def compute_question_distance(sorted_processed_blocks, tokenizer, all_conversations):
#     """
#     We assume the questions are asked at the end of all concatenated conversation blocks.
#     This function computes the distance of each question from the end to its corresponding conversation block.
#     Range of distance: [0, total_blocks-1]
#     """
#     total_blocks = len(sorted_processed_blocks)
#     all_qa = []
#
#
#
#     sorted_strings = []
#     for block in sorted_processed_blocks:
#         curr_block_string = ""
#         for item in block['conversation']:
#             curr_block_string += '\n' + item['content']
#         sorted_strings.append(curr_block_string)
#
#     for i, block in enumerate(sorted_processed_blocks):
#         # Only keep Q&As in the last block, i.e., the current session during conversation
#         if i + 1 < total_blocks:
#             continue
#
#         distance = total_blocks - (i + 1)
#         curr_block = sorted_strings[i]
#
#         # we assign distance to all qa in the current block
#         for idx, q in enumerate(block.get('qa', [])):
#             if not q:
#                 continue
#
#             # Get where the question will be asked
#             where = q['Where']
#             if where == 'END OF TEXT':
#                 block_num_q, start_index_q = i, len(curr_block) - 1 # This Q&A will be asked at the end of this block
#             else:
#                 block_num_q, start_index_q = utils.find_string_in_list(sorted_strings, where)
#
#             # count_tokens(" ".join([item['content'] for item in all_conversations if 'content' in item]), tokenizer, verbose=False)
#             num_tokens_q = sum([count_tokens(sorted_strings[i], tokenizer, verbose=False) for i in range(block_num_q)])
#             num_tokens_q += count_tokens(sorted_strings[block_num_q][:start_index_q], tokenizer, verbose=False)
#             curr_context = " ".join([sorted_strings[i] for i in range(block_num_q)]) + sorted_strings[block_num_q][:start_index_q]
#             # print('count_tokens(curr_context)', count_tokens(curr_context, tokenizer, verbose=False))
#
#             print('!!!!!!!', sum([count_tokens(sorted_strings[i], tokenizer, verbose=False) for i in range(block_num_q)]),
#                   count_tokens(" ".join([sorted_strings[i] for i in range(block_num_q)]), tokenizer, verbose=False),
#                   count_tokens(" ".join([item['content'] for item in sorted_processed_blocks]), tokenizer, verbose=False)
#             )
#
#             # Get where the reference information will be
#             if 'Reference' not in q:
#                 continue
#             else:
#                 if block['topic'] == 'writing' or block['topic'] == 'email' or block['topic'] == 'coding':
#                     # pick the first sentence
#                     block_num_ref, start_index_ref = i, 0
#                 elif 'Conversation' in q['Reference']:
#                     reference_event = q['Reference']['Conversation']
#                     reference_utterance = reference_event.split('\n')[1]
#                     block_num_ref, start_index_ref = utils.find_string_in_list(sorted_strings, reference_utterance)
#                 else:
#                     # For sequence of updates Q&A, it is a list of dictionary. We need to find the last one, i.e., the earliest one
#                     all_timestamps = [key for key in q['Reference'] if key != 'full_sequence']
#                     all_timestamps.sort(key=lambda x: datetime.strptime(x, "%m/%d/%Y"))
#                     reference_event = q['Reference'][all_timestamps[0]]['Conversation']
#                     # print('reference_event', reference_event)
#                     reference_utterance = reference_event.split('\n')[1]
#                     # print('reference_utterance', reference_utterance)
#                     block_num_ref, start_index_ref = utils.find_string_in_list(sorted_strings, reference_utterance)
#
#             num_tokens_ref = sum([count_tokens(sorted_strings[i], tokenizer, verbose=False) for i in range(block_num_ref)])
#             num_tokens_ref += count_tokens(sorted_strings[block_num_ref][:start_index_ref], tokenizer, verbose=False)
#
#             q['distance_blocks'] = block_num_q - block_num_ref
#             q['distance_tokens'] = num_tokens_q - num_tokens_ref + count_tokens(q['Question'], tokenizer, verbose=False)
#             q['context_length'] = num_tokens_q + count_tokens(q['Question'], tokenizer, verbose=False)
#             q['curr_context'] = curr_context
#             # print('len(curr_context)', len(curr_context), "context_length", q['context_length'])
#             all_qa.append(q)
#
#     return all_qa


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
        context_length = qa['context_length']
        curr_context = qa['curr_context']
        where = qa['Where'] if 'Where' in qa else None

        yield curr_context, question, formatted_question, correct_answer, all_options, distance_blocks, distance_tokens, question_type, topic, where, context_length


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--idx_persona', type=int, default=0, help='Index of the persona')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of conversation blocks')
    parser.add_argument('--format', type=str, default='string', help='Output conversation format: string or api_dict. Not applicable for qa')
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
    # all_strings = []
    new_content_samples = [{} for _ in range(len(chosen_blocks))]

    for block_idx, ((file_name, time_period), conversation) in enumerate(chosen_blocks):
        topic = file_name.split('_')[1]
        try:
            processed_conversation, latest_ts = process_conversation_block(topic, conversation, which_format)
        except Exception as e:
            print(f"{utils.Colors.FAIL}Error processing conversation block {file_name}{utils.Colors.ENDC}")
            continue

        qa = extract_qa(base_dir, topic, file_name, time_period)

        if topic == 'writing' or topic == 'email' or topic == 'coding':
            with open(os.path.join(args['inference']['output_dir'], 'writing', file_name), 'r') as file:
                data = json.load(file)
                original_sample = data.get("Original Sample")
                updated_sample = data.get("Updated Writing Sample")
            new_content_samples[block_idx] = {"Original Sample": original_sample, "Updated Sample": updated_sample}

        processed_blocks_dict[latest_ts] = {
            "conversation": processed_conversation[0],  # idx 0 corresponds to the conversation in the required format, either string or api_dict
            "file_name": file_name,
            "time_period": time_period,
            "last_timestamp": latest_ts,
            "topic": topic,
            "qa": qa
        }
        # all_strings.append(processed_conversation[-1]) # idx -1 always corresponds to the conversation in the plain string format

    # Topological sort chosen conversation blocks by the latest timestamp
    sorted_processed_blocks, sorted_strings, sorted_new_content_samples = topological_sort(processed_blocks_dict, new_content_samples, verbose)

    # Concatenate all conversation blocks
    all_conversations = concatenate_blocks(sorted_processed_blocks, sorted_new_content_samples, which_format, verbose)

    # Reiterate through all qa after block concatenations to add the distance information
    # total_num_tokens = sum([count_tokens(string, tokenizer, verbose=False) for string in all_strings])
    total_num_tokens = count_tokens(" ".join(item['content'] for item in all_conversations), tokenizer, verbose=False)
    if verbose:
        print(f"{utils.Colors.OKGREEN}Number of tokens: {total_num_tokens} on gpt-4o tokenizer{utils.Colors.ENDC}")
    all_qa = compute_question_distance(sorted_processed_blocks, sorted_strings, tokenizer, total_num_tokens)

    # Show all Q&As related to this concatenated conversation
    for curr_context, question, formatted_question, correct_answer, all_options, distance, question_type, topic, where, context_length in question_loader(all_qa):
        """
        The formatted_question is the input to the LLM model, and correct_answer is the target answer. 
        We (1) split the formatted_question (2) add the distance here, only for display purposes.
        Example usage: formatted_question -> LLM -> predicted_answer <-> correct_answer
        """
        question = formatted_question.split('\n', 1)[0]
        rest_of_qa = formatted_question[len(question):]

        print(f'{utils.Colors.OKGREEN}{question} [Distance {distance}] [Type {question_type}] [Topic {topic}] {utils.Colors.ENDC}{rest_of_qa}')
        print(f'Correct answer: {correct_answer}')