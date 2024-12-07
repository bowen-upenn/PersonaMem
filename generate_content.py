from tqdm import tqdm
import os
import numpy as np
import re
import json
import random
import torch

from query_llm import QueryLLM
import utils
import extract_conversation
from prepare_qa import process_json_from_api


def load_all_writing_sample(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    keys = list(data.keys())  # Preload the keys
    return data, keys


def extract_random_writing_sample(data, prompts):
    random_prompt = random.choice(prompts)
    curr_samples = data[random_prompt]
    return random.choice(curr_samples)


def convert_to_conversation(persona, writing_sample):
    # Rewrite the writing sample to be persona-aligned
    response = LLM.query_llm(step='new_content', data={'persona': persona, 'writing_sample': writing_sample}, action='rewrite_from_persona', verbose=False)
    response = process_json_from_api(response)
    writing_styles = response.get("Writing_styles", "")
    formatting_styles = response.get("Formatting_styles", "")
    updated_writing_sample = response.get("Updated_writing_sample", "")

    # Rewrite the persona-aligned writing sample as a conversation
    conversation = LLM.query_llm(step='new_content', action='rewrite_as_conversation', verbose=False)

    new_writing_sample = LLM.query_llm(step='new_content', data={'persona': persona, 'writing_styles': writing_styles, 'formatting_styles': formatting_styles}, action='write_new_sample', verbose=False)


if __name__ == '__main__':
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file', e)

    LLM = QueryLLM(args)

    # Load all JSON files
    json_data = []
    data_path = 'data/output'
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(data_path, filename)
            json_data.append(file_path)
    print(f"Loaded {len(json_data)} JSON files.")

    # Extract personas and conversations
    recommendation_data = {}
    for json_file in json_data:
        filename_parts = json_file.split('_')
        context = filename_parts[1]
        persona = extract_conversation.extract_persona(json_file)
        conversations = extract_conversation(json_file, context, which_conversation='all', which_format='string')

        # Query the LLM using prompts.prompt_for_recommendations() to get persona-oriented recommendations
        recommendation = LLM.query_llm(step='recommendation', content={"persona": persona, "conversations": conversations}, verbose=args['inference']['verbose'])

        # Save the recommendations and the corresponding JSON filename in a new JSON file
        recommendation_data[json_file] = recommendation

    # Randomly select three recommendations from the four different personas under the same context
    for json_file in json_data:
        filename_parts = json_file.split('_')
        context = filename_parts[1]
        persona_idx = filename_parts[2]
        persona_idx = int(persona_idx.replace('persona', ''))

        candidate_ids = [i for i in range(0, len(json_data)) if i != persona_idx]
        random_persona_ids = random.sample(candidate_ids, 3)
        random_sample_ids = [random.randint(0, 3) for _ in range(3)]

        recommendations_diff_personas = []
        for persona, sample_id in zip(random_persona_idx, random_sample_ids):
            # Construct the new file name for each randomly selected persona
            selected = f"conversation_{context}_persona{persona}_sample{sample_id}.json"
            selected = os.path.join(data_path, selected)
            recommendations_diff_persona = recommendation_data[selected]

    # Merge them as four MCQ options, and format a question like which recommendation best aligns with the persona's preferences

    # Repeat multiple times with different four sources

    # Save the question and the four MCQ options in each of the four original JSON file