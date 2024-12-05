import datetime
import json
import re
import yaml
import argparse
import ast
from sentence_transformers import SentenceTransformer, util
import random

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
    for block in history_blocks:
        for key, value in block.items():
            if key == timestamp:
                related_data.append(value)
    return related_data


def find_most_similar_event(SentenceBERT, side_note_sentence, related_data):
    """
    The same timestamp may have multiple events, like one in the general personal history and one in the contextual one.
    This function uses SentenceBERT to locate the single event we are actually targeting.
    """
    max_similarity = -1
    most_similar_data = None

    for data in related_data:
        event_sentence = data.get("event", "")
        similarity = util.pytorch_cos_sim(
            SentenceBERT.encode(side_note_sentence, convert_to_tensor=True),
            SentenceBERT.encode(event_sentence, convert_to_tensor=True)
        )
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_data = data

    return most_similar_data


def trace_event_history(timestamp, previous_blocks, verbose=False):
    """
    Traces the event history recursively, if needed, for knowledge updates.
    """
    linear_graph = {}
    while True:
        event_data = None
        for block in previous_blocks:
            event_data = block.get(timestamp)
            if event_data:
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


def generate_qa_static(seed_data, verbose):
    response = LLM.query_llm(step='qa_static', seed=seed_data, verbose=verbose)
    pass


def generate_qa_reasons_of_change(LLM, context, event_history):
    if context == "therapy":
        user = 'patient'
    elif context == 'legal':
        user = 'client'
    else:
        raise ValueError("Invalid context", context)

    qa_entries = []
    timestamps = list(event_history.keys())  # Get all timestamps in order

    for i, timestamp in enumerate(timestamps[:-1]):  # Iterate until the second-to-last timestamp
        current_details = event_history[timestamp]
        next_details = event_history[timestamps[i + 1]]
        reference = {
            timestamp: current_details,
            timestamps[i + 1]: next_details
        }

        if "[Reasons of Change]" in current_details:  # Only include events with reasons for change
            # Generate Q&A pairs on the reasons for change
            if "[Old Fact] Dislikes" in current_details:
                question = (
                    f"Why did the {user} {current_details['Event'].lower()[:-1]} in {timestamp} "
                    f"although the {user} dislikes {current_details['[Old Fact] Dislikes'].lower()}?"
                )
            elif "[Old Fact] Likes" in current_details:
                question = (
                    f"Why did the {user} {current_details['Event'].lower()[:-1]} in {timestamp} "
                    f"although the {user} likes {current_details['[Old Fact] Likes'].lower()}?"
                )
            else:
                continue
            correct_answer = current_details["[Reasons of Change]"]

            incorrect_answers = LLM.query_llm(step='qa_helper', data=f"Question: {question} Correct answer: {correct_answer}", action='propose_incorrect_reasons', verbose=False)
            match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
            if match:
                incorrect_answers = match.group(1)  # Extract the code block
            incorrect_answers = incorrect_answers.strip("```").strip()
            incorrect_answers = ast.literal_eval(incorrect_answers)
            # incorrect_answers = incorrect_answers.strip("```python").strip("```").strip()

            qa_entries.append({
                "Question": question,
                "Correct_Answer": correct_answer,
                "Incorrect_Answers": incorrect_answers,
                "Type": "reasons_of_change",
                "Reference": reference
            })

    # Save to JSON file
    print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
    print(json.dumps(qa_entries, indent=4))
    # with open(output_file, "w") as f:
    #     json.dump(qa_entries, f, indent=4)

    return qa_entries


def generate_qa_graph_of_updates(LLM, context, event_history):

    def _randomly_shorten_an_answer(incorrect_answer):
        # Randomly remove one event if the number of '->' is greater than 2
        events = [event.strip() for event in incorrect_answer.split('->')]
        random_event_index = random.randint(1, len(events) - 3)  # Exclude the first and last event

        # To remove one update step, we need to remove two consecutive events
        del events[random_event_index]
        del events[random_event_index + 1]

        incorrect_answer = ' -> '.join(events)
        return incorrect_answer

    if context == "therapy":
        user = "patient"
    elif context == "legal":
        user = "client"
    else:
        raise ValueError("Invalid context", context)

    """
    - Incorrect knowledge updates
    - Always dislikes
    - Always likes
    - Correct knowledge updates, but use the object never mentioned in the conversation
    - Incorrect knowledge updates, but use the object never mentioned in the conversation
    - Randomly substitute likes to dislikes (or vice versa) x 2
    - Miss some update steps, if the number of steps is greater than 2
    """
    correct_answer = ""
    incorrect_answers = ["" for _ in range(7)]
    timestamps = list(event_history.keys())  # Get all timestamps in order

    # Assert there are knowledge updates
    if len(timestamps) < 2:
        return

    # Based on what the user likes/dislikes, extract the parent object name to form the question, as well as random child objects for incorrect answers
    current_details = event_history[timestamps[0]]
    if "[Updated Fact] Likes" in current_details:
        data = current_details['[Updated Fact] Likes']
    else:
        data = current_details['[Updated Fact] Dislikes']
    response = LLM.query_llm(step='qa_helper', data=data, action='extract_object', verbose=False)

    response = response.strip("```json").strip("```python").strip("```").strip().replace("'", '"')
    response = json.loads(response)
    parent_object = response.get("parent_object", "")
    random_child_object = response.get("random_child_object", "")
    question = (
        f"How does the {user}'s preference towards {parent_object} evolve?"
    )

    # Iterate through the linear graph of updates to generate correct and incorrect answers
    for i, timestamp in enumerate(reversed(timestamps)):
        current_details = event_history[timestamp]

        if "[Updated Fact] Likes" in current_details or "[Fact] Likes" in current_details:
            curr_preference = current_details['[Updated Fact] Likes'].lower() if "[Updated Fact] Likes" in current_details else current_details['[Fact] Likes'].lower()
            if i == 0:
                correct_answer += f"The user likes {curr_preference}"
                incorrect_answers[0] = f"The user dislikes {curr_preference}"
                incorrect_answers[1] = f"The user always dislikes {curr_preference}"
                incorrect_answers[2] = f"The user always likes {curr_preference}"
                incorrect_answers[3] = f"The user likes {random_child_object}"
                incorrect_answers[4] = f"The user dislikes {random_child_object}"
                incorrect_answers[5] = f"The user {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"
                incorrect_answers[6] = f"The user {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"
            else:
                correct_answer += f" -> likes {curr_preference}"
                incorrect_answers[0] += f" -> dislikes {curr_preference}"
                incorrect_answers[3] += f" -> likes {random_child_object}"
                incorrect_answers[4] += f" -> dislikes {random_child_object}"
                incorrect_answers[5] += f" -> {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"
                incorrect_answers[6] += f" -> {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"

        elif "[Updated Fact] Dislikes" in current_details or "[Fact] Dislikes" in current_details:
            curr_preference = current_details['[Updated Fact] Dislikes'].lower() if "[Updated Fact] Dislikes" in current_details else current_details['[Fact] Dislikes'].lower()
            if i == 0:
                correct_answer += f"The user dislikes {curr_preference}"
                incorrect_answers[0] = f"The user likes {curr_preference}"
                incorrect_answers[1] = f"The user always dislikes {curr_preference}"
                incorrect_answers[2] = f"The user always likes {curr_preference}"
                incorrect_answers[3] = f"The user dislikes {random_child_object}"
                incorrect_answers[4] = f"The user likes {random_child_object}"
                incorrect_answers[5] = f"The user {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"
                incorrect_answers[6] = f"The user {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"
            else:
                correct_answer += f" -> dislikes {curr_preference}"
                incorrect_answers[0] += f" -> likes {curr_preference}"
                incorrect_answers[3] += f" -> dislikes {random_child_object}"
                incorrect_answers[4] += f" -> likes {random_child_object}"
                incorrect_answers[5] += f" -> {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"
                incorrect_answers[6] += f" -> {'likes' if random.choice([True, False]) else 'dislikes'} {curr_preference}"

    # Remove some random steps from the correct answer
    incorrect_answer_shorter = correct_answer[:]
    if incorrect_answer_shorter.count('->') > 3:
        incorrect_answers.append(_randomly_shorten_an_answer(incorrect_answer_shorter))

    # Remove repeated incorrect answers, if any resulted from the random process
    if incorrect_answers[5] == incorrect_answers[6]:
        del incorrect_answers[6]

    qa_entry = {
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "graph_of_updates",
        "Reference": event_history
    }

    # Save to JSON file
    print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
    print(json.dumps(qa_entry, indent=4))
    # with open(output_file, "w") as f:
    #     json.dump(qa_entry, f, indent=4)

    return qa_entry, parent_object


def generate_qa_recommendations(LLM, context, event_history, parent_object=None):
    if context == "therapy":
        user = "patient"
    elif context == "legal":
        user = "client"
    else:
        raise ValueError("Invalid context", context)

    timestamps = list(event_history.keys())  # Get all timestamps in order
    _, current_detail = timestamps[0], event_history[timestamps[0]]
    _, previous_detail = timestamps[1], event_history[timestamps[1]]
    
    if parent_object is None:
        if "[Updated Fact] Likes" in current_detail:
            data = current_detail['[Updated Fact] Likes']
        else:
            data = current_detail['[Updated Fact] Dislikes']
        response = LLM.query_llm(step='qa_helper', data=data, action='extract_object', verbose=False)
        response = response.strip("```json").strip("```python").strip("```").strip().replace("'", '"')
        response = json.loads(response)
        parent_object = response.get("parent_object", "")

    recent_two_events = json.dumps(current_detail, indent=4) + json.dumps(previous_detail, indent=4)
    response = LLM.query_llm(step='qa_helper', data={'user': user, 'parent_object': parent_object, 'events': recent_two_events}, action='recommendation', verbose=False)

    response = response.strip("```json").strip("```python").strip("```").strip()
    response = json.loads(response)
    question = response.get("Question", "")
    correct_answer = response.get("Answer", "")

    incorrect_answers = LLM.query_llm(step='qa_helper', data={'user': user, 'correct_answer': str(response)}, action='propose_incorrect_recommendations', verbose=False)
    match = re.search(r"```python\n(.*?)\n```", incorrect_answers, re.DOTALL)
    if match:
        incorrect_answers = match.group(1)  # Extract the code block
    incorrect_answers = incorrect_answers.strip("```").strip()
    incorrect_answers = ast.literal_eval(incorrect_answers)

    qa_entry = {
        "Question": question,
        "Correct_Answer": correct_answer,
        "Incorrect_Answers": incorrect_answers,
        "Type": "recommendations",
        "Reference": event_history
    }

    # Save to JSON file
    print(f'{utils.Colors.OKGREEN}Q&A:{utils.Colors.ENDC}')
    print(json.dumps(qa_entry, indent=4))
    # with open(output_file, "w") as f:
    #     json.dump(qa_entry, f, indent=4)

    return qa_entry



def process_conversation(action, LLM, SentenceBERT, conversation_key, data_path, verbose):
    # Load json file
    with open(data_path, 'r') as file:
        data = json.load(file)

    if 'therapy' in data_path:
        context = 'therapy'
    elif 'legal' in data_path:
        context = 'legal'
    else:
        raise ValueError("Invalid context", data_path)

    conversation = data.get(conversation_key, [])
    # Collect all side notes with timestamps in the current conversation
    side_notes = extract_side_notes_with_timestamps(conversation)

    history_keys = {
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

    previous_blocks = [data.get(key, {}) for key in history_keys.get(conversation_key, [])]

    for timestamp, side_note in side_notes:
        # Find related data in the previous personal history for each current event
        related_data = find_related_data(timestamp, previous_blocks)
        if not related_data:
            continue

        # If there are more than one related data with the same timestamp, find the single correct one
        if len(related_data) > 1:
            most_similar_data = find_most_similar_event(SentenceBERT, side_note, related_data)
        else:
            most_similar_data = related_data[0]

        # data_keys = [key.lower() for key in most_similar_data.keys()]
        if "Reasons of Change" in most_similar_data or "[Reasons of Change]" in most_similar_data:
            # Knowledge update
            event_history = trace_event_history(timestamp, previous_blocks, verbose=(action=='view_graphs'))
            # print(f"Knowledge update traced: {event_history}")
            if action == 'qa':
                qa_entries = generate_qa_reasons_of_change(LLM, context, event_history)
                qa_entry, parent_object = generate_qa_graph_of_updates(LLM, context, event_history)
                qa_entry = generate_qa_recommendations(LLM, context, event_history, parent_object=None)
        else:
            # Static knowledge point
            # print(f"Static knowledge point: {most_similar_data}")
            # generate_qa_static()
            pass


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--action', type=str, default="qa", help='Choose from qa, view_graphs')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    SentenceBERT = SentenceTransformer('all-MiniLM-L6-v2')
    LLM = QueryLLM(args)

    data_path = './data/output/conversation_therapy_persona0_sample0.json'
    process_conversation(cmd_args.action, LLM, SentenceBERT, conversation_key="Conversation Next Year", data_path=data_path, verbose=cmd_args.verbose)
