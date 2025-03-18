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
from json_repair import repair_json

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
        if isinstance(block, list):
            block = block[0]
        # print('timestamp', timestamp, 'block', block.keys())
        for key, value in block.items():
            if key == timestamp:
                related_data.append(value)
    # print('related_data', related_data)
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
                event_data['block_name'] = key
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


def generate_qa_factual(LLM, topic, event_history, random_event_histories=None, verbose=False):
    # Initialize a list to store the last two details with non-zero conversations
    qa_entries = []
    last_two_details = []
    timestamps = list(event_history.keys())

    # Iterate through the timestamps in reverse order
    for timestamp in timestamps:
        current_detail = event_history[timestamp]

        # Check if the conversation is non-zero
        if len(current_detail['Conversation']) > 0:
            last_two_details.append(current_detail.copy())

        # Stop once the list contains two details
        if len(last_two_details) == 2:
            break

    if len(last_two_details) == 1:
        last_two_details.insert(0, None)    # No more recent event

    # Find the last two events in this event graph, which should appear in two different conversation blocks.
    # The Q&As below will be based on the second last event, and will be asked immediately before the last event.
    # If the Q&As are asked immediately after the second last event, there won't be challenge for the model.
    conversation = last_two_details[1]['Conversation']
    conversation = list(conversation.split('\n'))
    user_utterance = conversation[-2]
    if "[Fact] Likes" in last_two_details[1] or "[Fact] Dislikes" in last_two_details[1]:
        related_fact = last_two_details[1]['[Fact] Likes'] if "[Fact] Likes" in last_two_details[1] else last_two_details[1]['[Fact] Dislikes']
    elif "[Updated Fact] Likes" in last_two_details[1] or "[Updated Fact] Dislikes" in last_two_details[1]:
        related_fact = last_two_details[1]['[Updated Fact] Likes'] if "[Updated Fact] Likes" in last_two_details[1] else last_two_details[1]['[Updated Fact] Dislikes']
    else:
        return qa_entries

    response = LLM.query_llm(step='qa_helper', data={'user_utterance': user_utterance, 'related_fact': related_fact}, action='recall_facts', verbose=False)
    response = utils.process_json_from_api(response)
    question = response.get("User Question", "")
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'response': correct_answer}, action='propose_incorrect_facts', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = repair_json(incorrect_answers)
    incorrect_answers = json.loads(incorrect_answers)
    # incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "recalling_facts_mentioned_by_the_user",
        "Topic": topic,
        "How_Many_Pref_Updates": len(timestamps),
        "Reference": last_two_details[1],
        "Where": last_two_details[0]["Conversation"].split('\n')[-2] if last_two_details[0] else "END OF TEXT"  # insert this question before this place, -2 to insert before the user's utterance
    })

    if random_event_histories:
        # Randomly sample three event histories
        random_event_histories = random.sample(random_event_histories, 3)
        random_event_histories = [f"({i + 1}) {history}" for i, history in enumerate(random_event_histories)]
        random_event_histories = "\n\n".join(random_event_histories)
        last_two_details[1]["other_previously_mentioned_events"] = random_event_histories

        response = LLM.query_llm(step='qa_helper', data={'event': user_utterance}, action='recall_facts_inverse', verbose=False)
        response = utils.process_json_from_api(response)
        question = response.get("User Question", "")
        correct_answer = response.get("Model Response", "")

        incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'random_event_histories': random_event_histories}, action='propose_incorrect_facts_inverse', verbose=False)
        match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
        if match:
            incorrect_answers = match.group(1)  # Extract the code block
        incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
        incorrect_answers = repair_json(incorrect_answers)
        incorrect_answers = json.loads(incorrect_answers)
        # incorrect_answers = ast.literal_eval(incorrect_answers)

        qa_entries.append({
            "Question": question,
            "Correct_Answer": correct_answer,
            "Incorrect_Answers": incorrect_answers,
            "Type": "identifying_new_things_not_mentioned_by_the_user",
            "Topic": topic,
            "How_Many_Pref_Updates": len(timestamps),
            "Reference": last_two_details[1],
            "Where": last_two_details[0]["Conversation"].split('\n')[-2] if last_two_details[0] else "END OF TEXT"  # insert this question before this place, -2 to insert before the user's utterance
        })

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


def generate_qa_reasons_of_change(LLM, topic, event_history, verbose=False):
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

    if len(last_two_details) < 2 or "[Reasons of Change]" not in last_two_details[0]:
        return qa_entries

    # The reason of change will be extracted from the most recent event
    # and queried immediately after the user utterance of this most recent event
    related_event = {
        "Event": last_two_details[0]["Event"],
        "[Reasons of Change]": last_two_details[0]["[Reasons of Change]"],
    }

    if "[Updated Fact] Likes" in last_two_details[0]:
        related_event["[Updated Fact] Likes"] = last_two_details[0]["[Updated Fact] Likes"]
        if "[Old Fact] Likes" in last_two_details[0]:
            return qa_entries
        related_event["[Old Fact] Dislikes"] = last_two_details[0]["[Old Fact] Dislikes"] if "[Old Fact] Dislikes" in last_two_details[0] else last_two_details[0]["[Fact] Dislikes"]
    else:
        related_event["[Updated Fact] Dislikes"] = last_two_details[0]["[Updated Fact] Dislikes"]
        if "[Old Fact] Dislikes" in last_two_details[0]:
            return qa_entries
        related_event["[Old Fact] Likes"] = last_two_details[0]["[Old Fact] Likes"] if "[Old Fact] Likes" in last_two_details[0] else last_two_details[0]["[Fact] Likes"]

    # This Q&A will be asked immediately before the last event
    response = LLM.query_llm(step='qa_helper', data={'event': str(related_event)}, action='generalize_reason_to_other_scenarios', verbose=False)
    response = utils.process_json_from_api(response)
    question = response.get("User Question", "")
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'user_utterance': question, 'reason_of_change': related_event["[Reasons of Change]"],
                                                              'model_response':correct_answer}, action='propose_incorrect_reasons_generalization', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = repair_json(incorrect_answers)
    incorrect_answers = json.loads(incorrect_answers)
    # incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "generalizing_past_reasons_in_memory_to_new_scenarios",
        "Topic": topic,
        "How_Many_Pref_Updates": len(timestamps),
        "Reference": last_two_details[0],
        "Where": "END OF TEXT"  # insert this question before this place
    })

    # This Q&A will be asked immediately after the user's utterance in the last event, but before the model's response
    response = LLM.query_llm(step='qa_helper', data={'event': str(related_event)}, action='ask_previous_reason_after_new_updates', verbose=False)
    response = utils.process_json_from_api(response)
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'response': correct_answer}, action='propose_incorrect_reasons_after_new_updates', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = repair_json(incorrect_answers)
    incorrect_answers = json.loads(incorrect_answers)
    # incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "recalling_the_reasons_behind_previous_updates",
        "Topic": topic,
        "How_Many_Pref_Updates": len(timestamps),
        "Reference": last_two_details[0],
        "Where": last_two_details[1]["Conversation"].split('\n')[-1]   # insert this question before this place, -1 to insert after the user's utterance
    })

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


def generate_qa_sequence_of_updates(LLM, topic, event_history, verbose=False):
    """
    This type of Q&A will be asked immediately after the user's utterance in the last event, but before the model's response.
    The model should follow up to the user by mentioning how the user's preference towards this thing or activity evolves.
    """
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
        if "[Fact] Likes" in last_event or "[Updated Fact] Likes" in last_event:    # there could be an init event without conversation, as a result, removed from the sequence
            full_sequence += f"The user likes {last_event['[Fact] Likes'].lower()}" if "[Fact] Likes" in last_event else f"The user likes {last_event['[Updated Fact] Likes'].lower()}"
        else:
            full_sequence += f"The user dislikes {last_event['[Fact] Dislikes'].lower()}" if "[Fact] Dislikes" in last_event else f"The user dislikes {last_event['[Updated Fact] Dislikes'].lower()}"

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
    incorrect_answers = repair_json(incorrect_answers)
    incorrect_answers = json.loads(incorrect_answers)
    # incorrect_answers = ast.literal_eval(incorrect_answers)

    colon_index = user_utterance.find(':')
    question = user_utterance[colon_index+1:].strip()# + " How do you think about it?"
    event_history['full_sequence'] = full_sequence
    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "tracking_the_full_sequence_of_preference_updates",
        "Topic": topic,
        "How_Many_Pref_Updates": len(timestamps),
        "Reference": event_history,
        "Where": last_event["Conversation"].split('\n')[-2]  # insert this question before this place, -2 since the question already includes the user's utterance
    })

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


def generate_qa_recommendations(LLM, topic, event_history, persona, parent_object=None, verbose=False):
    """
    Recommendation-type questions only care about the most recent two events for each conversation block
    """
    last_two_details = []
    timestamps = list(event_history.keys())

    # Iterate through the timestamps in reverse order
    for timestamp in timestamps:
        current_detail = event_history[timestamp]

        # Check if the conversation is non-zero
        # print('current_detail', current_detail)
        if 'Conversation' not in current_detail:
            continue
        if len(current_detail['Conversation']) > 0:
            last_two_details.append(current_detail.copy())

        # Stop once the list contains two details
        if len(last_two_details) == 2:
            break

    if len(last_two_details) == 1:
        last_two_details.insert(0, None)    # No more recent event

    if parent_object is None:
        # try:
        if "[Updated Fact] Likes" in last_two_details[1] or "[Fact] Likes" in last_two_details[1]:
            data = last_two_details[1]['[Updated Fact] Likes'] if "[Updated Fact] Likes" in last_two_details[1] else last_two_details[1]['[Fact] Likes']
        else:
            data = last_two_details[1]['[Updated Fact] Dislikes'] if "[Updated Fact] Dislikes" in last_two_details[1] else last_two_details[1]['[Fact] Dislikes']
        # except:
        #     return None
        response = LLM.query_llm(step='qa_helper', data=data, action='extract_object', verbose=False)
        response = utils.process_json_from_api(response)
        parent_object = response.get("parent_object", "")

    # Find the last two events in this event graph, which should appear in two different conversation blocks.
    # The Q&As below will be based on the second last event, and will be asked immediately before the last event.
    # If the Q&As are asked immediately after the second last event, there won't be challenge for the model.
    if "[Updated Fact] Likes" in last_two_details[1] or "[Fact] Likes" in last_two_details[1]:
        preference = last_two_details[1]['[Updated Fact] Likes'] if "[Updated Fact] Likes" in last_two_details[1] else last_two_details[1]['[Fact] Likes']
        preference = "Likes " + preference[0].lower() + preference[1:]
    else:
        preference = last_two_details[1]['[Updated Fact] Dislikes'] if "[Updated Fact] Dislikes" in last_two_details[1] else last_two_details[1]['[Fact] Dislikes']
        preference = "Dislikes " + preference[0].lower() + preference[1:]
    conversation = last_two_details[1]['Conversation']
    conversation = list(conversation.split('\n'))
    user_utterance = conversation[-2]

    response = LLM.query_llm(step='qa_helper', data={'parent_object':parent_object, 'preference': preference, 'user_utterance': user_utterance}, action='recommendation', verbose=False)
    response = utils.process_json_from_api(response)
    question = response.get("User Question", "")
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'question': question, 'model_response': correct_answer, 'preference': preference}, action='propose_incorrect_recommendations', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = repair_json(incorrect_answers)
    incorrect_answers = json.loads(incorrect_answers)
    # incorrect_answers = ast.literal_eval(incorrect_answers)

    identity = LLM.query_llm(step='qa_helper', data=persona["Expanded Persona"], action='extract_identity', verbose=False)
    stereotypical_answer = LLM.query_llm(step='qa_helper', data={'persona': identity, 'question': question, 'model_response': correct_answer}, action='propose_stereotypical_recommendation', verbose=False)
    incorrect_answers.append(stereotypical_answer)

    last_two_details[1]["identity"] = identity
    qa_entry = {
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "recommendation_aligned_with_users_latest_preferences",
        "Topic": topic,
        "How_Many_Pref_Updates": len(timestamps),
        "Reference": last_two_details[1],
        "Where": last_two_details[0]["Conversation"].split('\n')[-2] if last_two_details[0] else "END OF TEXT"  # insert this question before this place, -2 to insert before the user's utterance
    }

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entry, indent=4))

    return qa_entry


def generate_qa_recalling_preference(LLM, topic, event_history, verbose=False):
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

    if len(last_two_details) == 1:
        last_two_details.insert(0, None)  # No more recent event
        prev_event = {"Model_Response": None}
    else:
        prev_event = {"Model_Response": last_two_details[0]["Conversation"].split("\n")[2]}

    related_event = {
        "Event": last_two_details[1]["Event"],
        "User_Utterance": last_two_details[1]["Conversation"].split("\n")[1],
    }

    if "[Updated Fact] Likes" in last_two_details[1] or "[Fact] Likes" in last_two_details[1]:
        related_event['Preference'] = 'Likes' + last_two_details[1]['[Updated Fact] Likes'] if "[Updated Fact] Likes" in last_two_details[1] else 'Likes' + last_two_details[1]['[Fact] Likes']
    else:
        related_event['Preference'] = 'Dislikes' + last_two_details[1]['[Updated Fact] Dislikes'] if "[Updated Fact] Dislikes" in last_two_details[1] else 'Dislikes' + last_two_details[1]['[Fact] Dislikes']

    # This Q&A will be asked immediately before the last event
    response = LLM.query_llm(step='qa_helper', data={'Event': related_event['Event'], 'User_Utterance': related_event['User_Utterance'],
                                                     'Preference': related_event['Preference']}, action='recall_preference', verbose=False)
    response = utils.process_json_from_api(response)
    question = response.get("User Mention", "")
    correct_answer = response.get("Model Response", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'User_Mention': question, 'Response': correct_answer,
                                                              'Event': related_event['Event'], 'Old_Response': prev_event['Model_Response']}, action='propose_incorrect_preferences', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip().replace('\n', '')
    incorrect_answers = repair_json(incorrect_answers)
    incorrect_answers = json.loads(incorrect_answers)
    # incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entries.append({
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "recalling_the_latest_user_preferences",
        "Topic": topic,
        "How_Many_Pref_Updates": len(timestamps),
        "Reference": last_two_details[1],
        "Where": last_two_details[0]["Conversation"].split('\n')[-2] if last_two_details[0] else "END OF TEXT"  # insert this question before this place, -2 to insert before the user's utterance
    })

    # Save to JSON file
    if verbose:
        print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
        print(json.dumps(qa_entries, indent=4))

    return qa_entries


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
        "Type": "crafting_new_writing_samples_aligned_with_users_preferences",
        "Topic": "writing",
        "Where": "END OF TEXT",
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
    if 'writing' in data_path:
        topic = 'writing'
    elif 'coding' in data_path:
        topic = 'coding'
    elif 'email' in data_path:
        topic = 'email'
    else:
        raise ValueError("Invalid topic")

    source_data = utils.load_one_source_data(source_dir, all_source_files, topic=topic)

    # Load personas from the current file and three other random files
    assert len(all_writing_files) >= 4, "There should be at least 3 writing samples for comparison"
    all_data_paths = random.sample(['./data/output/'+topic+'/'+another_path for another_path in all_writing_files
                                    if another_path != data_path and another_path.rsplit('_', 1)[-2] != data_path.rsplit('_', 1)[-2]], 3)
    all_data_paths = [data_path] + all_data_paths   # The first one will be the correct answer

    persona = None
    new_writing_samples = []    # The first one will be the correct answer
    for i, curr_data_path in enumerate(all_data_paths):
        with open(curr_data_path, "r", encoding="utf-8") as f:
            curr_data = json.load(f)

        if topic == 'writing' or topic == 'email':
            preference = curr_data.get('Writing and Formatting Styles')
        elif topic == 'coding':
            preference = curr_data.get('Coding and Formatting Styles')
        else:
            raise ValueError("Invalid topic")
        new_writing_samples.append(LLM.query_llm(step='new_content', data={'topic': topic, 'persona': curr_data.get('Expanded Persona'), 'preferences': preference, 'source_data': source_data},
                                                 action='write_new_sample_oracle', verbose=False))
        if i == 0:
            persona = curr_data.get('Expanded Persona') + '\n\nWriting and Formatting Styles:\n\n' + preference

    qa_entry = {
        "Question": "Which of the following writing samples best aligns with the writer's persona above?",
        "Correct_Answer": new_writing_samples[0],
        "Incorrect_Answers": new_writing_samples[1:],
        "Type": "discriminating_new_writing_samples_aligned_with_users_preferences",
        "Topic": topic,
        "Where": "END OF TEXT",
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
    all_errored_path = []

    # # Generative type of QA
    # try:
    #     qa_entry = qa_generative(LLM, data, verbose)
    #     all_qa_entries.extend([qa_entry])
    # except Exception as e:
    #     print(f'{utils.Colors.FAIL}Error generating Q&A for generative type{utils.Colors.ENDC}{e}')

    # Discriminative type of QA
    try:
        for i in range(10):
            # print(f'{utils.Colors.OKGREEN}Generating Q&A No. {i}{utils.Colors.ENDC}')
            qa_entry = qa_discriminative(LLM, data_path, source_dir, all_source_files, all_writing_files, verbose)
            all_qa_entries.extend([qa_entry])
    except Exception as e:
        all_errored_path.append(f"Error generating Q&A for discriminative type {data_path}")
        print(f'{utils.Colors.FAIL}Error generating Q&A for discriminative type{utils.Colors.ENDC}{e}')

    # Save all Q&A entries to the JSON file at data_path
    # if "Q&A" not in data:
    data["Q&A"] = {'Conversation': all_qa_entries}
    # else:
    #     data["Q&A"]['Conversation'].extend(all_qa_entries)
    with open(data_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return all_errored_path


def evaluate_memory_from_conversation(action, LLM, SentenceBERT, conversation_key, data_path, clean, verbose):
    # Load json file
    with open(data_path, 'r') as file:
        data = json.load(file)
    print(f'{utils.Colors.OKGREEN}data_path: {data_path}: {conversation_key}{utils.Colors.ENDC}')

    all_errored_path = []
    all_empty_path = []

    # Remove old Q&A entries if they exist
    if clean and "Q&A" in data:
        if conversation_key in data["Q&A"]:
            # user_input = input("Are you sure to remove existing Q&As (y/n): ").strip().lower()
            # if user_input == 'y':
            del data["Q&A"][conversation_key]
            # Write the updated data back to the file
            with open(data_path, 'w') as file:
                json.dump(data, file, indent=4)
            print(f'{utils.Colors.WARNING}Removed old Q&As from the JSON file.{utils.Colors.ENDC}')
            # else:
            #     print("Skipping cleanup.")

    match = re.search(r'/([^/]+)/conversation_', data_path)
    topic = match.group(1)

    persona = {"Original Persona": data.get("Original Persona", ""), "Expanded Persona": data.get("Expanded Persona", "")}
    conversation = data.get(conversation_key, [])
    # Collect all side notes with timestamps in the current conversation
    side_notes = extract_side_notes_with_timestamps(conversation)

    previous_personal_histories = {
        "Init Conversation": ["Init Contextual Personal History"],
        "Conversation Next Week": ["Init Contextual Personal History", "Contextual Personal History Next Week"],
        "Conversation Next Month": ["Init Contextual Personal History", "Contextual Personal History Next Week",
                                    "Contextual Personal History Next Month"],
        "Conversation Next Year": ["Init Contextual Personal History", "Contextual Personal History Next Week",
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
        # print('related_data', related_data)
        if not related_data:
            continue

        # If there are more than one related data with the same timestamp, find the single correct one
        if len(related_data) > 1:
            corresponding_data = utils.find_most_similar_event(SentenceBERT, side_note, related_data)
        else:
            corresponding_data = related_data[0]

        event_history = trace_event_history(timestamp, previous_history_blocks, previous_conversation_blocks, verbose=(action=='view_graphs'))
        # print('event_history', event_history)

        if "Reasons of Change" in corresponding_data or "[Reasons of Change]" in corresponding_data:
            # Knowledge update
            if action == 'qa':
                try:
                    qa_entries = generate_qa_factual(LLM, topic, event_history, random_event_histories, verbose=verbose)
                    if len(qa_entries) > 0:
                        all_qa_entries.extend(qa_entries)
                except Exception as e:
                    all_errored_path.append(f"Error generating Q&A for static factual knowledge {data_path}:{conversation_key}")
                    print(f'{utils.Colors.FAIL}Error generating Q&A for static factual knowledge{utils.Colors.ENDC}{e}')
                try:
                    qa_entries = generate_qa_recalling_preference(LLM, topic, event_history, verbose=verbose)
                    if len(qa_entries) > 0:
                        all_qa_entries.extend(qa_entries)
                except Exception as e:
                    all_errored_path.append(f"Error generating Q&A for recalling preference {data_path}:{conversation_key}")
                    print(f'{utils.Colors.FAIL}Error generating Q&A for recalling preference{utils.Colors.ENDC}{e}')
                try:
                    qa_entries = generate_qa_reasons_of_change(LLM, topic, event_history, verbose=verbose)
                    if len(qa_entries) > 0:
                        all_qa_entries.extend(qa_entries)
                except Exception as e:
                    all_errored_path.append(f"Error generating Q&A for reasons of change {data_path}:{conversation_key}")
                    print(f'{utils.Colors.FAIL}Error generating Q&A for reasons of change{utils.Colors.ENDC}{e}')
                try:
                    qa_entries = generate_qa_sequence_of_updates(LLM, topic, event_history, verbose=verbose)
                    if len(qa_entries) > 0:
                        all_qa_entries.extend(qa_entries)
                except Exception as e:
                    all_errored_path.append(f"Error generating Q&A for sequence of updates {data_path}:{conversation_key}")
                    print(f'{utils.Colors.FAIL}Error generating Q&A for graph of updates{utils.Colors.ENDC}{e}')
                try:
                    qa_entry = generate_qa_recommendations(LLM, topic, event_history, persona, verbose=verbose)
                    if qa_entry:
                        all_qa_entries.extend([qa_entry])
                except Exception as e:
                    all_errored_path.append(f"Error generating Q&A for recommendations {data_path}:{conversation_key}")
                    print(f'{utils.Colors.FAIL}Error generating Q&A for recommendations{utils.Colors.ENDC}{e}')
        else:
            # pass
            try:
                qa_entries = generate_qa_factual(LLM, topic, event_history, verbose=verbose)
                if len(qa_entries) > 0:
                    all_qa_entries.extend(qa_entries)
            except Exception as e:
                all_errored_path.append(f"Error generating Q&A for static factual knowledge {data_path}:{conversation_key}")
                print(f'{utils.Colors.FAIL}Error generating Q&A for static factual knowledge{utils.Colors.ENDC}{e}')
            try:
                qa_entries = generate_qa_recalling_preference(LLM, topic, event_history, verbose=verbose)
                if len(qa_entries) > 0:
                    all_qa_entries.extend(qa_entries)
            except Exception as e:
                all_errored_path.append(f"Error generating Q&A for recalling preference {data_path}:{conversation_key}")
                print(f'{utils.Colors.FAIL}Error generating Q&A for recalling preference{utils.Colors.ENDC}{e}')
            try:
                qa_entry = generate_qa_recommendations(LLM, topic, event_history, persona, verbose=verbose)
                if qa_entry:
                    all_qa_entries.extend([qa_entry])
            except Exception as e:
                all_errored_path.append(f"Error generating Q&A for recommendations {data_path}:{conversation_key}")
                print(f'{utils.Colors.FAIL}Error generating Q&A for recommendations{utils.Colors.ENDC}{e}')

    # Save all Q&A entries to the JSON file at data_path
    if "Q&A" not in data:
        data["Q&A"] = {conversation_key: all_qa_entries}
    else:
        if conversation_key not in data["Q&A"]:
            data["Q&A"][conversation_key] = all_qa_entries
        else:
            data["Q&A"][conversation_key].extend(all_qa_entries)
    if len(all_qa_entries) == 0:
        all_empty_path.append(f"No Q&A entries generated for {data_path}: {conversation_key}")
        print(f'{utils.Colors.WARNING}No Q&A entries generated for {conversation_key}{utils.Colors.ENDC}')
    with open(data_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return all_errored_path, all_empty_path


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    torch.manual_seed(0)
    world_size = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if world_size > 1:
        assert world_size == 1

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--action', type=str, default="qa", help='Choose from qa, and view_graphs (not applicable for "writing" topic.')
    parser.add_argument('--topics', type=str, default="therapy", nargs="+",
                            help='Set conversation topics. Choose from therapy, legalConsultation, datingConsultation, foodRecommendation, onlineShopping, studyConsultation, '
                                 'travelPlanning, movieRecommendation, songRecommendation, homeDecoration, financialConsultation, healthConsultation, writing, email, coding.'
                                 'or all to select all existing topics under ./data/output/. '
                                 'If you want to select multiple topics manually, separate the names by space, e.g. --topics therapy legal'
                                 'Choose "irrelevant" if you want to generate data irrelevant to the topic to fill in long conversation context')  # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument('--n_persona', type=int, default=1, help='Set number of personas to generate')
    parser.add_argument('--n_samples', type=int, default=1, help='Set number of samples per topic to generate')
    parser.add_argument('--s_persona', type=int, default=0, help='Set the starting idx of personas to generate')
    parser.add_argument('--s_samples', type=int, default=0, help='Set the starting idx of samples per topic to generate')
    # parser.add_argument('--data', type=str, default="therapy_persona0_sample0", help='Path to the JSON data file')
    parser.add_argument('--time', type=str, default="next_year", help='Select the cut-off time (included) for the conversation data (not applicable for "writing" topic.')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    clean = False
    if cmd_args.clean:
        user_input = input("The 'clean' flag is set. Do you really want clean up all existing Q&As in the files (y/n): ").strip().lower()
        if user_input == 'y':
            clean = True
        else:
            print("Skipping cleanup.")
    # clean = cmd_args.clean

    LLM = QueryLLM(args)
    LLM.create_a_thread(step='qa')

    all_data_paths = []
    for topic in cmd_args.topics:
        for persona_id in range(cmd_args.s_persona, cmd_args.n_persona):
            data_path = f'./data/output/{topic}/conversation_{topic}_persona{persona_id}_sample0.json'
            all_data_paths.append(data_path)
    # match = re.match(r'^([^_]+)_', cmd_args.data)
    # data_path = './data/output/' + match.group(1) + '/conversation_' + cmd_args.data + '.json'

    if cmd_args.time == 'init':
        time_periods = ['Init Conversation']
    elif cmd_args.time == 'next_week':
        time_periods = ['Conversation Next Week']
    elif cmd_args.time == 'next_month':
        time_periods = ['Conversation Next Month']
    elif cmd_args.time == 'next_year':
        time_periods = ['Conversation Next Year']
    elif cmd_args.time == 'all':
        time_periods = ['Init Conversation', 'Conversation Next Week', 'Conversation Next Month', 'Conversation Next Year']
    else:
        raise ValueError("Invalid time", cmd_args.time)

    all_errored_paths = []
    all_empty_paths = []
    for data_path in all_data_paths:
        if 'writing' in data_path or 'coding' in data_path or 'email' in data_path:
            if 'writing' in data_path:
                source_dir = args['datasets']['writing_source_dir']
                all_source_files = utils.load_all_source_data(source_dir, 'writing')
                all_writing_files = utils.load_all_writing_data('writing')
            elif 'coding' in data_path:
                source_dir = args['datasets']['coding_source_dir']
                all_source_files = utils.load_all_source_data(source_dir, 'coding')
                all_writing_files = utils.load_all_writing_data('coding')
            elif 'email' in data_path:
                source_dir = args['datasets']['email_source_dir']
                all_source_files = utils.load_all_source_data(source_dir, 'email')
                all_writing_files = utils.load_all_writing_data('email')
            else:
                raise ValueError("Invalid topic")

            all_errored_path = evaluate_content_generation_from_memory(LLM, data_path=data_path, source_dir=source_dir, all_source_files=all_source_files, all_writing_files=all_writing_files, verbose=cmd_args.verbose)
            if len(all_errored_path) > 0:
                all_errored_paths.extend(all_errored_path)
        else:
            SentenceBERT = SentenceTransformer('all-MiniLM-L6-v2')
            for time_period in time_periods:
                all_errored_path, all_empty_path = evaluate_memory_from_conversation(cmd_args.action, LLM, SentenceBERT, conversation_key=time_period, data_path=data_path, clean=clean, verbose=cmd_args.verbose)
                if len(all_errored_path) > 0:
                    all_errored_paths.extend(all_errored_path)
                if len(all_empty_path) > 0:
                    all_empty_paths.extend(all_empty_path)

    print(f'{utils.Colors.FAIL}All errored paths:{utils.Colors.ENDC}')
    already_printed = []
    for path in all_errored_paths:
        if path not in already_printed:
            print(path)
        already_printed.append(path)
    print(f'{utils.Colors.WARNING}All empty paths:{utils.Colors.ENDC}')
    already_printed = []
    for path in all_empty_paths:
        if path not in already_printed:
            print(path)
    LLM.delete_a_thread(step='qa')
