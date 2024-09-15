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
            utils.append_json_to_file(args['datasets']['context'], output_file_path, curr_data_name='Context', parse_json=False)

            # Load a random source data in the given context and convert it from JSON to plain text
            source_data = utils.load_source_data(source_dir)
            context_conversation = utils.preprocess_source_data(source_data, args['datasets']['context'])

            # Send the conversation to the LLM as a background memory about the context
            LLM.create_a_thread()
            response = LLM.query_llm(step='source_data', content=context_conversation, verbose=args['inference']['verbose'])

            # Load a random persona
            random_row = random.choice(all_personas)
            persona = random_row.strip()[13:-2]  # Remove {"persona": "} and "}
            if args['inference']['verbose']:
                print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}{persona}')
            utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)

            # Expand the persona to at least five sentences
            persona = LLM.query_llm(step='expand_persona', content=persona, verbose=args['inference']['verbose'])
            utils.append_json_to_file(persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)

            # Generate personal history, conversations, and questions and answers in two consecutive turns
            steps = ['init_general_personal_history', 'init_contextual_personal_history', 'init_conversation', 'generate_questions', 'continue_conversation', 'second_expand', 'generate_questions', 'continue_conversation']
            data_names = ['General Personal History', 'Contextual Personal History', 'Initial Conversation', 'Initial Q&A Pairs', 'Initial Conversation', 'Second Expand', 'Expanded Q&A Pairs', 'Expanded Conversation']
            for step, data_name in zip(steps, data_names):
                content = None
                if step == 'init_general_personal_history':
                    content = persona
                response = LLM.query_llm(step=step, content=content, context=args['datasets']['context'], verbose=args['inference']['verbose'])
                utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True)
