import json
import os
import re
import argparse
import yaml
import torch
from sentence_transformers import SentenceTransformer
from query_llm import QueryLLM
import utils
from prepare_qa import evaluate_memory_from_conversation, evaluate_content_generation_from_memory, trace_event_history, extract_side_notes_with_timestamps

# TODO #1: adapt the prompts to this specific question type which ask about users' preference changes. Input and expected output for the prompt specification are below.
# Input: for the linear graph of every user, taken two concecutive events as the references.
# Output: if action == "preference_change": generate the correct response -- LLM should first acknowledge the older preference based on the [Old Fact] and [Old Event] part and then add something neutral to finish this utterance. 
# if action == "propose_incorrectpreference_change": generate three incorrect responses:
# Incorrect response 1. Opposite Acknowledgment of Old Preference -- If user previously like(dislike) something at [Old Event Date], LLM should first violate this old preference by saying eg. "I remember you dislike (like) the same thing" and then add something consistent with this wrong old preference to finish this utterance. 
# Incorrect response 2. Complete Forgetfulness -- If user previously like(dislike) something at [Old Event Date], LLM should first act as though the user never mentioned their preference by saying eg. Start directly by saying " It’s great that we’re discussing this now" without recalling previous [Old Fact] and then add something consistent with this ignorance of the old preference to finish this utterance
# Incorrect response 3. Repetition of Old (Opposite) Advice based on the old preference -- Maybe could direct quote previous LLM response to old preference at [Old Event Date] (but nor sure how this differs with personalized recommendation task, feel free to suggest better ideas -- happy to discuss furthur)

def prompts_for_preference_change(data, action):
    if action == "preference_change": # data['Reference'] refers to the two consecutive events
        prompt ="We want to evaluate whether a chatbot can remember user's preferences shared by the user during previous conversations, " \
                 "and whether the model can utilize its memory to provide a personalized response. Given this specific activity\n\n'" + data['Reference'] + "'\n\ndescribed by the user in a conversation with the chatbot:\n\n" + data['user_utterance'] + "\n\n" \
                 "The user has mentioned the detailed reason below of their preference update in previous conversations:\n\n" + data['event'] + "\n\n" \
                 "You should focus on the [Old Fact] and [Old Event] part. We actually want to evaluate if the model can remember and utilize this previous preference in the following conversation. " \
                 "Since the user has mentioned the preference in the previous session [Old Event Date]. " \
                 "Propose a response that specifically acknowledge the older preference based on the [Old Fact] and [Old Event] part the user mentioned on [Old Event Date] and then add something neutral to finish this utterance. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == "propose_incorrectpreference_change":
        prompt = "This is the correct personalized response to the question: " + data['question'] + ": " + data['response'] + "\n\n" \
                 "Please propose three incorrect options. Each incorrect response must violate the user’s stated preference in one or more of the following ways: " \
                 "1. Opposite Acknowledgment of Old Preference - first acknowledge the older preference based on the [Old Fact] and [Old Event] part and then add something neutral to finish this utterance"\
                 "2. Complete Forgetfulness- first act as though the user never mentioned their preference without recalling previous [Old Fact] and then add something consistent with this ignorance of the old preference to finish this utterance."\
                 "3. Repetition of Old (Opposite) Advice- repeat the old utterance by AI chatbot from a past conversation that was based on the opposite preference, ignoring the user’s updated or clarified preference."\
                 "Each option should share similar tone, matching length, and equal level of detail. Please do NOT be lazy! " \
                 "Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization." \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'
     
    return prompt

# TODO #2: Merge this new question type into the prepare_qa main generation function -- the following code should work for this question type i.e. loop over the entire event history and extract two exucutive events and query model to generate evaluation questions based on them
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
    #assert world_size == 1

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    #parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--action', type=str, default="qa", help='Choose from qa, and view_graphs (not applicable for "writing" context.')
    parser.add_argument('--data', type=str, default="therapy_persona0_sample0", help='Path to the JSON data file')
    parser.add_argument('--time', type=str, default="next_month", help='Select the cut-off time (included) for the conversation data (not applicable for "writing" context.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    #args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    #args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    #LLM = QueryLLM(cmd_args)
    #LLM.create_a_thread(step='qa')

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

    conversation_key=time_period
    print(f'{utils.Colors.OKGREEN}data_path: {data_path}: {conversation_key}{utils.Colors.ENDC}')
    #current_directory = os.getcwd()

    # Print the current working directory
    #print(f'Current working directory: {current_directory}')
    with open(data_path, 'r') as file:
        data = json.load(file)
    match = re.search(r'/([^/]+)/conversation_', data_path)
    context = match.group(1)
    # print("context")
    # print(context)
    persona = {"Original Persona": data.get("Original Persona", ""), "Expanded Persona": data.get("Expanded Persona", "")}
    # print("persona")
    # print(persona)
    conversation = data.get(conversation_key, [])
    # print("conversation")
    # print(type(conversation))
    # print(conversation[0])
    # Collect all side notes with timestamps in the current conversation
    side_notes = extract_side_notes_with_timestamps(conversation)
    #print(len(side_notes))
    #print(side_notes[0])

    # print("side_notes")
    # print(side_notes)
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

    all_conv_infill_entries = []

    for current_timestamp, side_note in side_notes:
        event_history = trace_event_history(current_timestamp, previous_history_blocks, previous_conversation_blocks, verbose=(cmd_args.action=='view_graphs'))

        all_timestamps = list(event_history.keys())
        if len(all_timestamps) == 1: # filter out events only mentioned once
            continue

        # select the current block from timestamp 
        # print(json.dumps(event_history, indent=4))
        # print(current_timestamp)
        # print(all_timestamps)
        cur_block = event_history[current_timestamp]
        
        # select the previous block
        prev_timestamp = all_timestamps[1]
        prev_block = event_history[prev_timestamp]
        reference = {
            "reference": {
                "cur_block": cur_block,
                "prev_block": prev_block
            }
        }
       
        # extract LLM turn from current block
        correct_llm_turn = cur_block["Conversation"].split("\n")[-1]
        distractor_llm_turn = prev_block["Conversation"].split("\n")[-1]
        

        # print(prev_block["Conversation"])
        
        # Query LLM
        # TODO #3: append this new question type to all q&a's
        correct_answer = "<tba>"
        distractor_answer = ["<tba>", "<tba>", "<tba>"]
        dummy_question = "<question>"

        # append the question to all_qa_entries
        all_conv_infill_entries.append({
            "side_note": side_note,
            "Reference": reference,
            "question": dummy_question,
            "correct_answer": correct_answer,
            "distractor_answer": distractor_answer
        })

    # After looping over all side notes, save the conversation infill entries
    if "Q&A" not in data:
        data["Q&A"] = {}

    data["Q&A"]["conversation_infill"] = all_conv_infill_entries
    
    output_path = data_path.replace(".json", "_infill.json")
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f'{utils.Colors.OKGREEN}Saved conversation infill entries to {output_path}{utils.Colors.ENDC}')

# TODO #3: Add the generative evaluation methods for comparing joint probabilities.


