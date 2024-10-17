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
        for idx_persona in range(int(args['inference']['num_personas'])):
            # Load a random persona
            random_row = random.choice(all_personas)
            persona = random_row.strip()[13:-2]  # Remove prefix '{"persona":' and suffix '"}'
            if args['inference']['verbose']:
                print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}{persona}')

            # Expand the persona to at least five sentences
            expanded_persona = LLM.query_llm(step='expand_persona', content=persona, verbose=args['inference']['verbose'])

            for idx_context in range(int(args['inference']['num_contexts_per_persona'])):
                if int(args['inference']['num_contexts_per_persona']) == 1:
                    curr_context = args['datasets']['curr_context']
                else:
                    curr_context = args['datasets']['all_contexts'][idx_context]

                for idx_sample in range(int(args['inference']['num_samples_per_context'])):
                    output_file_path = os.path.join(args['inference']['output_dir'], f'{args["inference"]["output_file_name"]}_{curr_context}_persona{idx_persona}_sample{idx_sample}.json')
                    utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                    utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                    utils.append_json_to_file(curr_context, output_file_path, curr_data_name='Context', parse_json=False)

                    LLM.create_a_thread()

                    # Load a random conversation history from the chosen real-world dataset
                    if curr_context == 'therapy':
                        source_dir = args['datasets']['therapy_source_dir']
                    else:
                        raise NotImplementedError("Unknown context: {}".format(curr_context))
                    utils.append_json_to_file(curr_context, output_file_path, curr_data_name='Context', parse_json=False)

                    # Load a random source data to the LLM as a background memory about the context
                    source_data = utils.load_source_data(source_dir)
                    context_conversation = utils.preprocess_source_data(source_data, curr_context)
                    _ = LLM.query_llm(step='source_data', content=context_conversation, verbose=args['inference']['verbose'])

                    # Generate personal history, conversations, and questions and answers in two consecutive turns
                    steps = ['init_general_personal_history', 'init_contextual_personal_history', 'init_conversation', 'generate_questions', 'expand_history_and_conversation', 'generate_questions']
                    data_names = ['General Personal History', 'Contextual Personal History', 'Initial Conversation', 'Initial Q&A Pairs', 'Expand History and Conversation', 'Expanded Q&A Pairs']
                    for step, data_name in zip(steps, data_names):
                        content = None
                        if step == 'init_general_personal_history':
                            if idx_context > 0:
                                utils.append_json_to_file(LLM.init_general_personal_history, output_file_path, curr_data_name=data_name, parse_json=True)
                                continue    # only generate general personal history once, to be shared across multiple contexts for the same persona
                            content = expanded_persona

                        response = LLM.query_llm(step=step, content=content, context=curr_context, idx_context=idx_context, verbose=args['inference']['verbose'])

                        if step == 'expand_history_and_conversation' and idx_context > 0:
                            utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True, expanded_general_personal_history=LLM.expanded_general_personal_history)
                        else:
                            utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True)
