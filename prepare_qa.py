import datetime
import json
import re
import yaml
import argparse
import ast
from sentence_transformers import SentenceTransformer, util
import random
from tqdm import tqdm
import torch

import utils
from query_llm import QueryLLM


def extract_side_notes_with_timestamps(conversation):
    """
    Extracts Side_Notes with timestamps from a conversation.
    """
    text_pattern = r'\b\[?Side[\s_]?Notes?\]?\b'
    timestamp_pattern = r"\b\d{2}/\d{2}/\d{4}\b"

    # Extract lines with 'Side_Nodes' and their timestamps
    filtered_lines = [line for line in conversation if re.search(text_pattern, line)]
    timestamps = [re.search(timestamp_pattern, line).group() for line in filtered_lines if re.search(timestamp_pattern, line)]
    return [(timestamp, line) for timestamp, line in zip(timestamps, filtered_lines)]


def find_related_data(timestamp, history_blocks):
    """
    Finds events in the provided history blocks that match the timestamp.
    """
    related_data = []
    for _, block in history_blocks.items():
        for key, value in block.items():
            if key == timestamp:
                related_data.append(value)
    return related_data


def get_time_period_from_block_name(bname):
    if "Next Year" in bname:
        return "Conversation Next Year"
    elif "Next Month" in bname:
        return "Conversation Next Month"
    elif "Next Week" in bname:
        return "Conversation Next Week"
    else:
        return "Init Conversation"


def trace_event_history(timestamp, previous_history_blocks, previous_conversation_blocks, verbose=False):
    """
    Traces the event history recursively for knowledge updates.
    """
    linear_graph = {}

    while True:
        event_data = None
        for key, block in previous_history_blocks.items():
            event_data = block.get(timestamp)

            # If found the matched time stamp in the block
            if event_data:
                # Map each personal history block to a conversation
                time_period = get_time_period_from_block_name(key)
                conversation = previous_conversation_blocks[time_period]
                found = False
                for i, line in enumerate(conversation):
                    if line.startswith("Side_Note") and timestamp in line:
                        event_data['Conversation'] = conversation[i] + '\n' + conversation[i + 1] + '\n' + conversation[i + 2]
                        found = True
                        break
                if not found:
                    event_data['Conversation'] = ""
                break

        if not event_data:
            break  # No further history to trace
        # print('event_data', timestamp, event_data)

        linear_graph[timestamp] = event_data
        if "Old Event" in event_data or "[Old Event]" in event_data:
            # Get the timestamp of the old event
            old_event_timestamp = event_data.get("Old Event Date") or event_data.get("[Old Event Date]", "")
            # print('old_event_timestamp', old_event_timestamp)

            # Update timestamp for next iteration
            timestamp = old_event_timestamp
        else:
            break  # No further history to trace

    if verbose:
        print(f'{utils.Colors.OKGREEN}linear_graph:{utils.Colors.ENDC}')
        print(json.dumps(linear_graph, indent=4))
    return linear_graph


def generate_qa_static_factual(LLM, context, event_history, random_event_histories=None, verbose=False):
    if context == "therapy":
        user = 'patient'
    elif context == 'legal':
        user = 'client'
    else:
        user = 'user'

    # Initialize a list to store the last two details with non-zero conversations
    qa_entries = []
    last_two_details = []
    timestamps = list(event_history.keys())

    # Iterate through the timestamps in reverse order
    for timestamp in timestamps:
        current_detail = event_history[timestamp]

        # Check if the conversation is non-zero
        if len(current_detail['Conversation']) > 0:
            last_two_details.append(current_detail)

        # Stop once the list contains two details
        if len(last_two_details) == 2:
            break

    # Find the last two events in this event graph, which should appear in two different conversation blocks.
    # The Q&As below will be based on the second last event, and will be asked immediately before the last event.
    # If the Q&As are asked immediately after the second last event, there won't be challenge for the model.
    conversation = last_two_details[1]['Conversation']
    conversation = list(conversation.split('\n'))
    user_utterance = conversation[-2]

    response = LLM.query_llm(step='qa_helper', data={'user': user, 'event': user_utterance}, action='recall_facts', verbose=False)
    response = utils.process_json_from_api(response)
    question = response.get("User Question", "")
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'response': correct_answer}, action='propose_incorrect_facts', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "factual",
        "Context": context,
        "Reference": last_two_details[1]
    })

    if random_event_histories:
        # Randomly sample three event histories
        random_event_histories = random.sample(random_event_histories, 3)
        random_event_histories = [f"({i + 1}) {history}" for i, history in enumerate(random_event_histories)]
        random_event_histories = "\n\n".join(random_event_histories)
        last_two_details[1]["other_previously_mentioned_events"] = random_event_histories

        response = LLM.query_llm(step='qa_helper', data={'user': user, 'event': user_utterance}, action='recall_facts_inverse', verbose=False)
        response = utils.process_json_from_api(response)
        question = response.get("User Question", "")
        correct_answer = response.get("Model Response", "")

        incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'random_event_histories': random_event_histories}, action='propose_incorrect_facts_inverse', verbose=False)
        match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
        if match:
            incorrect_answers = match.group(1)  # Extract the code block
        incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
        incorrect_answers = ast.literal_eval(incorrect_answers)

        qa_entries.append({
            "Question": question,
            "Correct_Answer": correct_answer,
            "Incorrect_Answers": incorrect_answers,
            "Type": "factual_inverse",
            "Context": context,
            "Reference": last_two_details[1]
        })

    # for i, timestamp in enumerate(timestamps):
    #     current_detail = event_history[timestamp]
    #     next_detail = event_history[timestamps[i + 1]]
    #
    #     # Avoid any events not mentioned in the conversation
    #     if len(current_detail['Conversation']) == 0:
    #         continue
    #
    #     print('current_detail', current_detail)
    #
    #     response = LLM.query_llm(step='qa_helper', data={'user': user, 'event': current_detail['Event']}, action='factual_qa', verbose=False)
    #     response = utils.process_json_from_api(response)
    #     question = response.get("Question", "")
    #     correct_answer = response.get("Answer", "")
    #
    #     incorrect_answers = LLM.query_llm(step='qa_helper', data=str(response), action='propose_incorrect_facts', verbose=False)
    #     match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    #     if match:
    #         incorrect_answers = match.group(1)  # Extract the code block
    #     incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    #     incorrect_answers = ast.literal_eval(incorrect_answers)
    #
    #     qa_entries.append({
    #         "Question": question,
    #         "Correct_Answer": correct_answer,
    #         "Incorrect_Answers": incorrect_answers,
    #         "Type": "static_factual",
    #         "Context": context,
    #         "Reference": event_history[current_timestamp]
    #     })
    #
    #     abstention_response = LLM.query_llm(step='qa_helper', data=question, action='abstention', verbose=False)
    #     abstention_response = utils.process_json_from_api(abstention_response)
    #     abstention_question = abstention_response.get("New Question", "")
    #     abstention_object = abstention_response.get("New Name", "")
    #
    #     incorrect_abstention_answers = [correct_answer]
    #     random_idn = random.sample(range(0, len(incorrect_answers)), 2)
    #     for i in random_idn:
    #         incorrect_abstention_answers.append(incorrect_answers[i])
    #
    #     qa_entries.append({
    #         "Question": abstention_question,
    #         "Correct_Answer": "Haven't mentioned " + abstention_object + " in the conversation",
    #         "Incorrect_Answers": incorrect_abstention_answers,
    #         "Type": "abstention",
    #         "Context": context,
    #         "Reference": event_history[current_timestamp]
    #     })
    #     break

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


def generate_qa_reasons_of_change(LLM, context, event_history, verbose=False):
    if context == "therapy":
        user = 'patient'
    elif context == 'legal':
        user = 'client'
    else:
        user = 'user'

    # Initialize a list to store the last two details with non-zero conversations
    qa_entries = []
    last_two_details = []
    timestamps = list(event_history.keys())

    # Iterate through the timestamps in reverse order
    for timestamp in timestamps:
        current_detail = event_history[timestamp]

        # Check if the conversation is non-zero
        if len(current_detail['Conversation']) > 0:
            last_two_details.append(current_detail)

        # Stop once the list contains two details
        if len(last_two_details) == 2:
            break

    # Find the last two events in this event graph, which should appear in two different conversation blocks.
    # The Q&As below will be based on the second last event, and will be asked immediately before the last event.
    # If the Q&As are asked immediately after the second last event, there won't be challenge for the model.
    related_event = {
        "Event": last_two_details[1]["Event"],
        "[Reasons of Change]": last_two_details[1]["[Reasons of Change]"],
    }
    if "[Updated Fact] Likes" in last_two_details[1]:
        related_event["[Updated Fact] Likes"] = last_two_details[1]["[Updated Fact] Likes"]
        related_event["[Old Fact] Dislikes"] = last_two_details[1]["[Old Fact] Dislikes"]
    else:
        related_event["[Updated Fact] Dislikes"] = last_two_details[1]["[Updated Fact] Dislikes"]
        related_event["[Old Fact] Likes"] = last_two_details[1]["[Old Fact] Likes"]

    # This Q&A will be asked immediately before the last event
    response = LLM.query_llm(step='qa_helper', data={'user': user, 'event': str(related_event)}, action='generalize_reason_to_other_scenarios', verbose=False)
    response = utils.process_json_from_api(response)
    question = response.get("User Utterance", "")
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'user_utterance': question, 'reason_of_change': related_event["[Reasons of Change]"],
                                                              'model_response':correct_answer}, action='propose_incorrect_reasons_generalization', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "reasons_of_change_generalization",
        "Context": context,
        "Reference": last_two_details[1]
    })

    # This Q&A will be asked immediately after the user's utterance in the last event, but before the model's response
    response = LLM.query_llm(step='qa_helper', data={'user': user, 'event': str(related_event)}, action='ask_previous_reason_after_new_updates', verbose=False)
    response = utils.process_json_from_api(response)
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'response': correct_answer}, action='propose_incorrect_reasons_after_new_updates', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "reasons_of_change_after_new_updates",
        "Context": context,
        "Reference": last_two_details[1]
    })

    # for i, timestamp in enumerate(timestamps[:-1]):  # Iterate until the second-to-last timestamp
    #     current_details = event_history[timestamp]
    #     next_details = event_history[timestamps[i + 1]]
    #
    #     # Avoid any events not mentioned in the conversation
    #     if len(current_details['Conversation']) == 0 or len(next_details['Conversation']) == 0:
    #         continue
    #
    #     reference = {
    #         timestamp: current_details,
    #         timestamps[i + 1]: next_details
    #     }
    #
    #     if "[Reasons of Change]" in current_details:  # Only include events with reasons for change
    #         # Generate Q&A pairs on the reasons for change
    #         if "[Old Fact] Dislikes" in current_details:
    #             question = (
    #                 f"Why did the {user} {current_details['Event'].lower()[:-1]} in {timestamp} "
    #                 f"although the {user} dislikes {current_details['[Old Fact] Dislikes'].lower()}?"
    #             )
    #         elif "[Old Fact] Likes" in current_details:
    #             question = (
    #                 f"Why did the {user} {current_details['Event'].lower()[:-1]} in {timestamp} "
    #                 f"although the {user} likes {current_details['[Old Fact] Likes'].lower()}?"
    #             )
    #         else:
    #             continue
    #         correct_answer = current_details["[Reasons of Change]"]
    #
    #         incorrect_answers = LLM.query_llm(step='qa_helper', data=f"Question: {question} Correct answer: {correct_answer}", action='propose_incorrect_reasons', verbose=False)
    #         match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    #         if match:
    #             incorrect_answers = match.group(1)  # Extract the code block
    #         incorrect_answers = incorrect_answers.strip("```").strip()
    #         incorrect_answers = ast.literal_eval(incorrect_answers)
    #         # incorrect_answers = incorrect_answers.strip("```python").strip("```").strip()
    #
    #         qa_entries.append({
    #             "Question": question,
    #             "Correct_Answer": correct_answer,
    #             "Incorrect_Answers": incorrect_answers,
    #             "Type": "reasons_of_change",
    #             "Context": context,
    #             "Reference": reference
    #         })

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


def generate_qa_graph_of_updates(LLM, context, event_history, verbose=False):
    """
    This type of Q&A will be asked immediately after the user's utterance in the last event, but before the model's response.
    The model should follow up to the user by mentioning how the user's preference towards this thing or activity evolves.
    """
    if context == "therapy":
        user = "patient"
    elif context == "legal":
        user = "client"
    else:
        user = 'user'

    qa_entries = []
    full_sequence = ""
    previous_details = None
    timestamps = list(event_history.keys())  # Get all timestamps in order

    # Find the most recent event
    last_event = None
    for timestamp in timestamps:
        current_detail = event_history[timestamp]

        # Check if the conversation is non-zero
        if len(current_detail['Conversation']) > 0:
            last_event = current_detail
            break
    user_utterance = last_event['Conversation'].split('\n')[-2]

    if len(timestamps) == 1:
        if "[Fact] Likes" in last_event:
            full_sequence += f"The user likes {last_event['[Fact] Likes'].lower()}"
        else:
            full_sequence += f"The user dislikes {last_event['[Fact] Dislikes'].lower()}"

    for i, timestamp in enumerate(reversed(timestamps)):
        current_details = event_history[timestamp]

        # Avoid any events not mentioned in the conversation
        if len(current_details['Conversation']) == 0:
            continue

        if "[Updated Fact] Likes" in current_details or "[Fact] Likes" in current_details:
            if previous_details is not None and ("[Updated Fact] Likes" in previous_details or "[Fact] Likes" in previous_details):
                continue    # No need to repeat two consecutive preferences without changes
            curr_preference = current_details['[Updated Fact] Likes'].lower() if "[Updated Fact] Likes" in current_details else current_details['[Fact] Likes'].lower()
            full_sequence += f"The user likes {curr_preference}" if len(full_sequence) == 0 else f" -> likes {curr_preference}"

        elif "[Updated Fact] Dislikes" in current_details or "[Fact] Dislikes" in current_details:
            if previous_details is not None and ("[Updated Fact] Dislikes" in previous_details or "[Fact] Dislikes" in previous_details):
                continue
            curr_preference = current_details['[Updated Fact] Dislikes'].lower() if "[Updated Fact] Dislikes" in current_details else current_details['[Fact] Dislikes'].lower()
            full_sequence += f"The user dislikes {curr_preference}" if len(full_sequence) == 0 else f" -> dislikes {curr_preference}"

        previous_details = current_details

    # This Q&A will be asked immediately before the last event
    response = LLM.query_llm(step='qa_helper', data={'full_sequence': full_sequence, 'user_utterance': user_utterance}, action='recall_sequence', verbose=False)
    response = utils.process_json_from_api(response)
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'model_response': correct_answer}, action='propose_incorrect_sequence', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = ast.literal_eval(incorrect_answers)

    question = user_utterance + " How do you think about it?"
    event_history['full_sequence'] = full_sequence
    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "sequence_of_updates",
        "Context": context,
        "Reference": event_history
    })
    #
    # def _randomly_shorten_an_answer(incorrect_answer, remove_last_event=True):
    #     # Randomly remove one event if the number of '->' is greater than 2
    #     events = [event.strip() for event in incorrect_answer.split('->')]
    #     if remove_last_event:
    #         random_event_index = len(events) - 1
    #     else:
    #         random_event_index = random.randint(1, len(events) - 3)  # Exclude the first and last event
    #
    #     # To remove one update step, we need to remove two consecutive events
    #     del events[random_event_index]
    #     if not remove_last_event:
    #         del events[random_event_index + 1]
    #
    #     incorrect_answer = ' -> '.join(events)
    #     return incorrect_answer
    #
    # if context == "therapy":
    #     user = "patient"
    # elif context == "legal":
    #     user = "client"
    # else:
    #     user = 'user'
    #
    # """
    # - Correct answer type I
    # - Incorrect answers type I
    #     - Same as the correct knowledge updates, except for missing the last one or adding an additional one
    #     - Always dislikes
    #     - Always likes
    #
    # - Correct answer type II
    #     - The user has never mentioned this object
    # - Incorrect answers type II
    #     - Same as the correct knowledge updates, except for missing the last one or adding an additional one, but use the object never mentioned in the conversation
    #     - Correct knowledge updates, but use the object never mentioned in the conversation
    #     - Incorrect knowledge updates, but use the object never mentioned in the conversation
    # """
    # qa_entries = []
    # correct_answer = ["", ""]
    # incorrect_answers = [["", "", ""], ["", "", ""]]
    # timestamps = list(event_history.keys())  # Get all timestamps in order
    #
    # # Assert there are knowledge updates
    # if len(timestamps) < 2:
    #     return None, None
    #
    # # Based on what the user likes/dislikes, extract the parent object name to form the question, as well as random child objects for incorrect answers
    # current_details = event_history[timestamps[0]]
    # if "[Updated Fact] Likes" in current_details:
    #     data = current_details['[Updated Fact] Likes']
    # else:
    #     data = current_details['[Updated Fact] Dislikes']
    # response = LLM.query_llm(step='qa_helper', data=data, action='extract_object', verbose=False)
    #
    # response = utils.process_json_from_api(response)
    # parent_object = response.get("parent_object", "")
    # random_child_object = response.get("random_child_object", "")
    # question = (
    #     f"How does the {user}'s preference towards {parent_object} evolve?"
    # )
    #
    # # Iterate through the linear graph of updates to generate correct and incorrect answers
    # previous_details = None
    # final_preference = None
    # event_count = 0
    # for i, timestamp in enumerate(reversed(timestamps)):
    #     current_details = event_history[timestamp]
    #
    #     # Avoid any events not mentioned in the conversation
    #     if len(current_details['Conversation']) == 0:
    #         continue
    #
    #     if "[Updated Fact] Likes" in current_details or "[Fact] Likes" in current_details:
    #         if previous_details is not None and ("[Updated Fact] Likes" in previous_details or "[Fact] Likes" in previous_details):
    #             continue  # We encountered a missing event not mentioned in the conversation, so there is no preference update reflected in the conversation itself
    #         if i + 1 == len(timestamps):
    #             final_preference = 'likes'
    #         event_count += 1
    #
    #         curr_preference = current_details['[Updated Fact] Likes'].lower() if "[Updated Fact] Likes" in current_details else current_details['[Fact] Likes'].lower()
    #         if len(correct_answer[0]) == 0:
    #             correct_answer[0] += f"The user likes {curr_preference}"
    #             incorrect_answers[0][0] = f"The user likes {curr_preference}"  # will perturb later on
    #             incorrect_answers[0][1] = f"The user always dislikes {curr_preference}"
    #             incorrect_answers[0][2] = f"The user always likes {curr_preference}"
    #
    #             correct_answer[1] += f"The user has never mentioned {random_child_object}"
    #             incorrect_answers[1][0] = f"The user likes {random_child_object}"  # will perturb later on
    #             incorrect_answers[1][1] = f"The user likes {random_child_object}"
    #             incorrect_answers[1][2] = f"The user dislikes {random_child_object}"
    #         else:
    #             correct_answer[0] += f" -> likes {curr_preference}"
    #             incorrect_answers[0][0] += f" -> likes {curr_preference}"  # will perturb later on
    #
    #             incorrect_answers[1][0] += f" -> likes {random_child_object}"
    #             incorrect_answers[1][1] += f" -> likes {random_child_object}"  # will perturb later on
    #             incorrect_answers[1][2] += f" -> dislikes {random_child_object}"
    #
    #     elif "[Updated Fact] Dislikes" in current_details or "[Fact] Dislikes" in current_details:
    #         if previous_details is not None and ("[Updated Fact] Dislikes" in previous_details or "[Fact] Dislikes" in previous_details):
    #             continue
    #         if i + 1 == len(timestamps):
    #             final_preference = 'dislikes'
    #         event_count += 1
    #
    #         curr_preference = current_details['[Updated Fact] Dislikes'].lower() if "[Updated Fact] Dislikes" in current_details else current_details['[Fact] Dislikes'].lower()
    #         if len(correct_answer[0]) == 0:
    #             correct_answer[0] += f"The user dislikes {curr_preference}"
    #             incorrect_answers[0][0] = f"The user dislikes {curr_preference}"  # will perturb later on
    #             incorrect_answers[0][1] = f"The user always dislikes {curr_preference}"
    #             incorrect_answers[0][2] = f"The user always likes {curr_preference}"
    #
    #             correct_answer[1] += f"The user has never mentioned {random_child_object}"
    #             incorrect_answers[1][0] = f"The user dislikes {curr_preference}"   # will perturb later on
    #             incorrect_answers[1][1] = f"The user dislikes {random_child_object}"
    #             incorrect_answers[1][2] = f"The user likes {random_child_object}"
    #         else:
    #             correct_answer[0] += f" -> dislikes {curr_preference}"
    #             incorrect_answers[0][0] += f" -> dislikes {curr_preference}"   # will perturb later on
    #
    #             incorrect_answers[1][0] += f" -> dislikes {random_child_object}"
    #             incorrect_answers[1][1] += f" -> dislikes {random_child_object}"   # will perturb later on
    #             incorrect_answers[1][2] += f" -> likes {random_child_object}"
    #
    #     previous_details = current_details
    #
    # if event_count == 1:
    #     return None, None
    # elif event_count <= 2:  # The only way is to add additional events not mentioned in the conversation
    #     if final_preference == 'likes':
    #         incorrect_answers[0][0] += f" -> dislikes {curr_preference}"
    #         incorrect_answers[1][0] += f" -> dislikes {random_child_object}"
    #     elif final_preference == 'dislikes':
    #         incorrect_answers[0][0] += f" -> likes {curr_preference}"
    #         incorrect_answers[1][0] += f" -> likes {random_child_object}"
    # else:
    #     if random.random() > 0.5:
    #         if final_preference == 'likes':
    #             incorrect_answers[0][0] += f" -> dislikes {curr_preference}"
    #             incorrect_answers[1][0] += f" -> dislikes {random_child_object}"
    #         elif final_preference == 'dislikes':
    #             incorrect_answers[0][0] += f" -> likes {curr_preference}"
    #             incorrect_answers[1][0] += f" -> likes {random_child_object}"
    #     else:   # Remove the last related event mentioned in the conversation
    #         incorrect_answers[0][0] = _randomly_shorten_an_answer(incorrect_answers[0][0], remove_last_event=True)
    #         incorrect_answers[1][0] = _randomly_shorten_an_answer(incorrect_answers[1][0], remove_last_event=True)
    #
    # qa_entries.append({
    #     "Question": question,
    #     "Correct_Answer": correct_answer[0],
    #     "Incorrect_Answers": incorrect_answers[0],
    #     "Type": "graph_of_updates",
    #     "Context": context
    # })
    # qa_entries.append({
    #     "Question": question,
    #     "Correct_Answer": correct_answer[1],
    #     "Incorrect_Answers": incorrect_answers[1],
    #     "Type": "graph_of_updates_abstention",
    #     "Context": context
    # })
    # qa_entries.append({"Reference": event_history})

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


def generate_qa_recommendations(LLM, context, event_history, persona, parent_object=None, verbose=False):
    """
    Recommendation-type questions only care about the most recent two events for each conversation block
    """
    if context == "therapy":
        user = "patient"
    elif context == "legal":
        user = "client"
    else:
        user = 'user'

    timestamps = list(event_history.keys())  # Get all timestamps in order
    _, current_detail = timestamps[0], event_history[timestamps[0]]
    
    if parent_object is None:
        if "[Updated Fact] Likes" in current_detail:
            data = current_detail['[Updated Fact] Likes']
        else:
            data = current_detail['[Updated Fact] Dislikes']
        response = LLM.query_llm(step='qa_helper', data=data, action='extract_object', verbose=False)
        response = utils.process_json_from_api(response)
        parent_object = response.get("parent_object", "")

    # Avoid any events not mentioned in the conversation, while it is safe to assume that the first one must has been mentioned
    recent_two_events = json.dumps(current_detail, indent=4)
    reference = {timestamps[0]: current_detail}

    if len(timestamps) > 1:
        _, previous_detail = timestamps[1], event_history[timestamps[1]]
        if len(previous_detail['Conversation']) > 0:
            recent_two_events += json.dumps(previous_detail, indent=4)
            reference[timestamps[1]] = previous_detail
    response = LLM.query_llm(step='qa_helper', data={'user': user, 'parent_object': parent_object, 'events': recent_two_events}, action='recommendation', verbose=False)

    response = utils.process_json_from_api(response)
    question = response.get("Question", "")
    correct_answer = response.get("Answer", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'user': user, 'qa': str(response)}, action='propose_incorrect_recommendations', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip()
    incorrect_answers = ast.literal_eval(incorrect_answers)

    identity = LLM.query_llm(step='qa_helper', data=persona["Expanded Persona"], action='extract_identity', verbose=False)
    stereotypical_answer = LLM.query_llm(step='qa_helper', data={'user': user, 'persona': identity+". "+persona["Original Persona"], 'qa': str(response)}, action='propose_stereotype_recommendation', verbose=False)
    incorrect_answers.append(stereotypical_answer)

    qa_entry = {
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "recommendations",
        "Context": context,
        "Reference": reference
    }

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entry, indent=4))

    return qa_entry

# # TODO
# def generate_qa_personalized_response(LLM, context, event_history, verbose=False):
#     if context == "therapy":
#         user = 'patient'
#     elif context == 'legal':
#         user = 'client'
#     else:
#         user = 'user'
#
#     qa_entries = []
#     timestamps = list(event_history.keys())  # Get all timestamps in order
#
#     for i, current_timestamp in enumerate(timestamps):
#         current_detail = event_history[current_timestamp]
#
#         """
#         If the user likes sth
#           - (1) The model should first acknowledge that and then propose sth new, right before the next turn
#           - (2) The model should first acknowledge that and then propose a recommendation that aligns with the like, right before the next section
#         If the user does not like sth in this section
#           - (1) The model should first acknowledge that and then try to persuade the user to like it, right before the next section
#           - (2) The model should first acknowledge that and then propose a recommendation that aligns with the dislike, right before the next section
#
#         We define each section as a group of three utterance: Side_Note, user, and agent.
#         We define each block as a group of conversation sections within the same period of time.
#
#         Having a response right after the user's utterance in this section would make the problem too easy, so we shall ask the question right before the next section.
#         This question will be challenging if the next section sits in the next conversation block, and there are blocks of other topics in between.
#
#         We need to store the location of the next section, and in the inference, we cut this conversation block at that location and ask the question.
#         This can be achieved by storing the beginning sentence of the next section, and in the inference time after block concatenations, we use regular expression to find the location.
#         """
#
#         # Avoid any events not mentioned in the conversation
#         if len(current_detail['Conversation']) == 0:
#             continue
#
#         print('current_detail', current_detail, '\n')
#
#         # utterances = current_detail['Conversation'].split("\n")
#         # user_utterance = utterances[1]
#         # agent_utterance = utterances[2]
#         #
#         # response = LLM.query_llm(step='qa_helper', data={'user': user, 'event': str(current_detail), 'user_utterance': user_utterance, 'agent_utterance': agent_utterance}, action='personalized_response', verbose=False)
#         # response = utils.process_json_from_api(response)
#         # question = response.get("Question", "")
#         # correct_answer = response.get("Answer", "")
#         #
#         # incorrect_answers = LLM.query_llm(step='qa_helper', data={'user': user, 'prev_response': response}, action='propose_other_responses', verbose=False)
#         # match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
#         # if match:
#         #     incorrect_answers = match.group(1)  # Extract the code block
#         # incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
#         # incorrect_answers = ast.literal_eval(incorrect_answers)
#         #
#         # qa_entries.append({
#         #     "Question": question,
#         #     "Correct_Answer": correct_answer,
#         #     "Incorrect_Answers": incorrect_answers,
#         #     "Type": "personalized_responses",
#         #     "Context": context,
#         #     "Reference": event_history[current_timestamp]
#         # })
#
#     # Save to JSON file
#     if verbose:
#         print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
#         print(json.dumps(qa_entries, indent=4))
#
#     return qa_entries


def qa_generative(LLM, curr_data, verbose=False):
    # Write new content that aligns and violates the persona
    conversation = curr_data.get("Conversation", [])
    conversation = utils.remove_side_notes(conversation)
    conversation = "\n".join(conversation)

    new_writing_sample = LLM.query_llm(step='new_content', data=conversation, action='write_new_sample', verbose=verbose)
    violated_writing_styles = LLM.query_llm(step='new_content', data=conversation, action='write_violating_sample', verbose=verbose)

    # Evaluate the new contents
    persona = curr_data.get("Expanded Persona", "")
    preferences = curr_data.get("Writing and Formatting Styles", "")
    aligned_results = LLM.query_llm(step='eval_new_content', data={'persona': persona, 'preferences': preferences,
                                                                   'paragraph1': new_writing_sample, 'paragraph2': violated_writing_styles}, action='evaluate_aligned', verbose=verbose)
    violated_results = LLM.query_llm(step='eval_new_content', action='evaluate_violated', verbose=verbose)

    qa_entry = {
        "Aligned_new_writing_sample": new_writing_sample,
        "Violated_new_writing_styles": violated_writing_styles,
        "Aligned_results": aligned_results,
        "Violated_results": violated_results,
        "Type": "new_content_generative",
        "Context": "writing",
        "Reference": preferences,
    }

    if verbose:
        print(f'{utils.Colors.OKGREEN}Aligned Results:{utils.Colors.ENDC}')
        print(json.dumps(aligned_results, indent=4))
        print(f'{utils.Colors.OKGREEN}Violated Results:{utils.Colors.ENDC}')
        print(json.dumps(violated_results, indent=4))

    return qa_entry


def qa_discriminative(LLM, data_path, source_dir, all_source_files, all_writing_files, verbose=False):
    # Load a random creative writing sample
    source_data = utils.load_one_source_data(source_dir, all_source_files, context='writing')

    # Load personas from the current file and three other random files
    assert len(all_writing_files) >= 4, "There should be at least 3 writing samples for comparison"
    all_data_paths = random.sample(['./data/output/writing/'+another_path for another_path in all_writing_files
                                    if another_path != data_path and another_path.rsplit('_', 1)[-2] != data_path.rsplit('_', 1)[-2]], 3)
    all_data_paths = [data_path] + all_data_paths   # The first one will be the correct answer

    persona = None
    new_writing_samples = []    # The first one will be the correct answer
    for i, curr_data_path in enumerate(all_data_paths):
        with open(curr_data_path, "r", encoding="utf-8") as f:
            curr_data = json.load(f)

        new_writing_samples.append(LLM.query_llm(step='new_content', data={'persona': curr_data.get('Expanded Persona'), 'preferences': curr_data.get('Writing and Formatting Styles'), 'source_data': source_data},
                                                 action='write_new_sample_oracle', verbose=False))
        if i == 0:
            persona = curr_data.get('Expanded Persona') + '\n\nWriting and Formatting Styles:\n\n' + curr_data.get('Writing and Formatting Styles')

    qa_entry = {
        "Question": "Which of the following writing samples best aligns with the writer's persona above?",
        "Correct_Answer": new_writing_samples[0],
        "Incorrect_Answers": new_writing_samples[1:],
        "Type": "new_content_discriminative",
        "Context": "writing",
        "Reference": persona,
    }

    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entry, indent=4))

    return qa_entry


def evaluate_content_generation_from_memory(LLM, data_path, source_dir, all_source_files, all_writing_files, verbose):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_qa_entries = []

    # # Generative type of QA
    # qa_entry = qa_generative(LLM, data, verbose)
    # all_qa_entries.extend([qa_entry])

    # Discriminative type of QA
    qa_entry = qa_discriminative(LLM, data_path, source_dir, all_source_files, all_writing_files, verbose)
    all_qa_entries.extend([qa_entry])

    # Save all Q&A entries to the JSON file at data_path
    if "Q&A" not in data:
        data["Q&A"] = {'Conversation': all_qa_entries}
    else:
        data["Q&A"]['Conversation'].extend(all_qa_entries)
    with open(data_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return


def evaluate_memory_from_conversation(action, LLM, SentenceBERT, conversation_key, data_path, verbose):
    # Load json file
    with open(data_path, 'r') as file:
        data = json.load(file)
    print(f'{utils.Colors.OKGREEN}data_path: {data_path}: {conversation_key}{utils.Colors.ENDC}')

    # Remove old Q&A entries if they exist
    if "Q&A" in data:
        if conversation_key in data["Q&A"]:
            del data["Q&A"][conversation_key]
            # Write the updated data back to the file
            with open(data_path, 'w') as file:
                json.dump(data, file, indent=4)
            print(f'{utils.Colors.WARNING}Removed old Q&As from the JSON file.{utils.Colors.ENDC}')

    match = re.search(r'/([^/]+)/conversation_', data_path)
    context = match.group(1)

    persona = {"Original Persona": data.get("Original Persona", ""), "Expanded Persona": data.get("Expanded Persona", "")}
    conversation = data.get(conversation_key, [])
    # Collect all side notes with timestamps in the current conversation
    side_notes = extract_side_notes_with_timestamps(conversation)

    previous_personal_histories = {
        "Init Conversation": ["Init General Personal History", "Init Contextual Personal History"],
        "Conversation Next Week": ["Init General Personal History", "General Personal History Next Week",
                                   "Init Contextual Personal History", "Contextual Personal History Next Week"],
        "Conversation Next Month": ["Init General Personal History", "General Personal History Next Week",
                                    "General Personal History Next Month",
                                    "Init Contextual Personal History", "Contextual Personal History Next Week",
                                    "Contextual Personal History Next Month"],
        "Conversation Next Year": ["Init General Personal History", "General Personal History Next Week",
                                   "General Personal History Next Month", "General Personal History Next Year",
                                   "Init Contextual Personal History", "Contextual Personal History Next Week",
                                   "Contextual Personal History Next Month", "Contextual Personal History Next Year"]
    }
    previous_conversations = {
        "Init Conversation": ["Init Conversation"],
        "Conversation Next Week": ["Init Conversation", "Conversation Next Week"],
        "Conversation Next Month": ["Init Conversation", "Conversation Next Week", "Conversation Next Month"],
        "Conversation Next Year": ["Init Conversation", "Conversation Next Week", "Conversation Next Month", "Conversation Next Year"]
    }

    previous_history_blocks = {key: data.get(key, {}) for key in previous_personal_histories.get(conversation_key, [])}
    previous_conversation_blocks = {key: data.get(key, []) for key in previous_conversations.get(conversation_key, [])}
    all_qa_entries = []


    # sample some random event histories
    random_event_histories = []
    for timestamp, side_note in side_notes:
        # Find related data in the previous personal history for each current event
        related_data = find_related_data(timestamp, previous_history_blocks)
        if not related_data:
            continue

        event_history = trace_event_history(timestamp, previous_history_blocks, previous_conversation_blocks, verbose=(action=='view_graphs'))
        timestamps = list(event_history.keys())
        last_event = event_history[timestamps[0]]
        if len(last_event['Conversation']) == 0:
            continue

        user_utterance = last_event['Conversation'].split("\n")[-2]
        random_event_histories.append(user_utterance)


    # prepare Q&A entries
    for timestamp, side_note in side_notes:
        # Find related data in the previous personal history for each current event
        related_data = find_related_data(timestamp, previous_history_blocks)
        if not related_data:
            continue

        # If there are more than one related data with the same timestamp, find the single correct one
        if len(related_data) > 1:
            corresponding_data = utils.find_most_similar_event(SentenceBERT, side_note, related_data)
        else:
            corresponding_data = related_data[0]

        event_history = trace_event_history(timestamp, previous_history_blocks, previous_conversation_blocks, verbose=(action=='view_graphs'))

        if "Reasons of Change" in corresponding_data or "[Reasons of Change]" in corresponding_data:
            # Knowledge update
            if action == 'qa':
                # try:
                # qa_entries = generate_qa_static_factual(LLM, context, event_history, random_event_histories, verbose=verbose)
                # all_qa_entries.extend(qa_entries)
                # except:
                #     print(f'{utils.Colors.FAIL}Error generating Q&A for static factual knowledge{utils.Colors.ENDC}')
                # try:
                # qa_entries = generate_qa_reasons_of_change(LLM, context, event_history, verbose=verbose)
                # all_qa_entries.extend(qa_entries)
                # except:
                #     print(f'{utils.Colors.FAIL}Error generating Q&A for reasons of change{utils.Colors.ENDC}')
                # parent_object = None
                # try:
                qa_entries = generate_qa_graph_of_updates(LLM, context, event_history, verbose=verbose)
                if qa_entries is not None:
                    all_qa_entries.extend(qa_entries)
                # except:
                #     print(f'{utils.Colors.FAIL}Error generating Q&A for graph of updates{utils.Colors.ENDC}')
                # try:
                #     qa_entry = generate_qa_recommendations(LLM, context, event_history, persona, parent_object, verbose=verbose)
                #     all_qa_entries.extend([qa_entry])
                # except:
                #     print(f'{utils.Colors.FAIL}Error generating Q&A for recommendations{utils.Colors.ENDC}')
                # # qa_entries = generate_qa_personalized_response(LLM, context, event_history, verbose=verbose)
                # # all_qa_entries.extend(qa_entries)
        else:
            pass
            # Static knowledge point
            # try:
            #     qa_entries = generate_qa_static_factual(LLM, context, event_history, verbose=verbose)
            #     all_qa_entries.extend(qa_entries)
            # except:
            #     print(f'{utils.Colors.FAIL}Error generating Q&A for static factual knowledge{utils.Colors.ENDC}')

    # Save all Q&A entries to the JSON file at data_path
    if "Q&A" not in data:
        data["Q&A"] = {conversation_key: all_qa_entries}
    else:
        if conversation_key not in data["Q&A"]:
            data["Q&A"][conversation_key] = all_qa_entries
        else:
            data["Q&A"][conversation_key].extend(all_qa_entries)
    with open(data_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return


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

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--action', type=str, default="qa", help='Choose from qa, and view_graphs (not applicable for "writing" context.')
    parser.add_argument('--data', type=str, default="therapy_persona0_sample0", help='Path to the JSON data file')
    parser.add_argument('--time', type=str, default="next_year", help='Select the cut-off time (included) for the conversation data (not applicable for "writing" context.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    LLM = QueryLLM(args)
    LLM.create_a_thread(step='qa')

    match = re.match(r'^([^_]+)_', cmd_args.data)
    data_path = './data/output/' + match.group(1) + '/conversation_' + cmd_args.data + '.json'

    if cmd_args.time == 'init':
        time_period = 'Init Conversation'
    elif cmd_args.time == 'next_week':
        time_period = 'Conversation Next Week'
    elif cmd_args.time == 'next_month':
        time_period = 'Conversation Next Month'
    elif cmd_args.time == 'next_year':
        time_period = 'Conversation Next Year'
    else:
        raise ValueError("Invalid time", cmd_args.time)

    # try:
    if 'writing' in data_path:
        source_dir = args['datasets']['writing_source_dir']
        all_source_files = utils.load_all_source_data(source_dir, 'writing')
        all_writing_files = utils.load_all_writing_data()
        evaluate_content_generation_from_memory(LLM, data_path=data_path, source_dir=source_dir, all_source_files=all_source_files, all_writing_files=all_writing_files, verbose=cmd_args.verbose)
    else:
        SentenceBERT = SentenceTransformer('all-MiniLM-L6-v2')
        evaluate_memory_from_conversation(cmd_args.action, LLM, SentenceBERT, conversation_key=time_period, data_path=data_path, verbose=cmd_args.verbose)
    # except Exception as e:
    #     print(f'{utils.Colors.FAIL}Error processing {data_path}: {e}{utils.Colors.ENDC}')
