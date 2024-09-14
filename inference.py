from tqdm import tqdm
import os
import numpy as np
import re
import json
import random
import torch

from query_llm import QueryLLM
import utils


def inference(args):
    # Load all personas
    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()
    LLM = QueryLLM(args)

    with torch.no_grad():
        for idx_sample in range(int(args['inference']['num_samples'])):
            output_file_path = os.path.join(args['inference']['output_dir'], f'{args["inference"]["output_file_name"]}_{idx_sample}.json')

            # Load a random conversation history from the chosen real-world dataset
            if args['datasets']['context'] == 'therapy':
                source_dir = args['datasets']['therapy_source_dir']
            else:
                raise NotImplementedError("Unknown context: {}".format(args['datasets']['context']))

            # Load a random source data in the given context and convert it from JSON to plain text
            source_data = utils.load_source_data(source_dir)
            context_conversation = utils.preprocess_source_data(source_data, args['datasets']['context'])

            # # Send the conversation to the LLM as a background memory about the context
            LLM.create_a_thread()
            response = LLM.query_llm(step='source_data', content=context_conversation, verbose=args['inference']['verbose'])

            # Load a random persona
            random_row = random.choice(all_personas)
            persona = random_row.strip()[13:-2]  # Remove {"persona": "} and "}
            if args['inference']['verbose']:
                print(f'{utils.Colors.OKGREEN}{"Persona"}:{utils.Colors.ENDC}{persona}')
            utils.append_json_to_file(persona, output_file_path, curr_data_name='Persona', parse_json=False)
            utils.append_json_to_file(args['datasets']['context'], output_file_path, curr_data_name='Context', parse_json=False)

            # Expand the persona to personal history
            response = LLM.query_llm(step='expand_persona', content=persona, verbose=args['inference']['verbose'])
            utils.append_json_to_file(response, output_file_path, curr_data_name='General Personal History', parse_json=True)

            # Expand the persona and personal history to conversation
            response = LLM.query_llm(step='init_conversation', context=args['datasets']['context'], verbose=args['inference']['verbose'])
            utils.append_json_to_file(response, output_file_path, curr_data_name='Initial Conversation', parse_json=True)

            # Generate a list of memory-related questions and answers
            response = LLM.query_llm(step='generate_questions', verbose=args['inference']['verbose'])
            utils.append_json_to_file(response, output_file_path, curr_data_name='Initial Q&A Pairs', parse_json=True)

            # Continue writing new personal history with conflicts and another conversation
            response = LLM.query_llm(step='second_expand', context=args['datasets']['context'], verbose=args['inference']['verbose'])
            utils.append_json_to_file(response, output_file_path, curr_data_name='Second Expand', parse_json=True)

            # Continue generating another list of memory-related questions and answers
            response = LLM.query_llm(step='generate_questions', verbose=args['inference']['verbose'])
            utils.append_json_to_file(response, output_file_path, curr_data_name='Expanded Q&A Pairs', parse_json=True)
