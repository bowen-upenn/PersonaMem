from tqdm import tqdm
import os
import numpy as np
import re
import json
import random
import torch
import ast

from query_llm import QueryLLM
import utils


def inference(args):
    # Load all personas
    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()
    LLM = QueryLLM(args)

    for idx_persona in range(int(args['inference']['start_persona_idx']), int(args['inference']['num_personas'])):
        # Load a random persona
        found = utils.find_existing_persona_files(idx_persona)
        if found:
            # Ensure that every data file with the same idx_persona share the same persona
            persona, expanded_persona, start_time = found['persona'], found['expanded_persona'], found['start_time']
            LLM.expanded_persona = expanded_persona
            if args['inference']['verbose']:
                print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}')
                print(persona)
                print(f'{utils.Colors.OKGREEN}{"Expanded Persona"}:{utils.Colors.ENDC}')
                print(expanded_persona)
        else:
            # Create a new persona for the new idx_persona
            random_row = random.choice(all_personas)
            persona = random_row.strip()[13:-2]  # Remove prefix '{"persona":' and suffix '"}'
            if args['inference']['verbose']:
                print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}{persona}')

            # Expand the persona to at least five sentences
            start_time = utils.pick_a_random_time()
            expanded_persona = LLM.query_llm(step='expand_persona', persona=persona, start_time=start_time, verbose=args['inference']['verbose'])

        # Clean up the names of contexts
        all_contexts = [ctx.strip() for ctx in args['datasets']['context']]

        # Since we assign a consecutive time frame for all contexts, we randomly permute contexts to ensure generalization
        if len(all_contexts) > 1:
            random.shuffle(all_contexts)

        # Loop through each context in the list
        for idx_context, curr_context in enumerate(all_contexts):
            # Process each context as needed
            print(f'{utils.Colors.OKGREEN}Processing context: {curr_context}, {idx_context+1}/{len(all_contexts)}{utils.Colors.ENDC}')

            # Load a random conversation history from the chosen real-world dataset
            if curr_context == 'writing':
                source_dir = args['datasets']['writing_source_dir']
            elif curr_context == 'legal':
                source_dir = args['datasets']['legal_source_dir']
            elif curr_context == 'therapy':
                source_dir = args['datasets']['therapy_source_dir']
            else:
                raise NotImplementedError("Unknown context: {}".format(curr_context))
            all_source_files = utils.load_all_source_data(source_dir, curr_context)

            # Set a consecutive time frame for different contexts for each persona, while all samples below are independent
            if idx_context > 0:
                start_time = utils.pick_a_random_time_within_a_year(start_time)

            for idx_sample in range(int(args['inference']['start_sample_idx']), int(args['inference']['num_samples_per_context'])):
                output_file_path = os.path.join(args['inference']['output_dir'], os.path.join(f'{curr_context}', f'{args["inference"]["output_file_name"]}_{curr_context}_persona{idx_persona}_sample{idx_sample}.json'))
                utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                utils.append_json_to_file(curr_context, output_file_path, curr_data_name='Context', parse_json=False)
                print(f'{utils.Colors.OKGREEN}Output file path: {output_file_path}{utils.Colors.ENDC}')

                LLM.create_a_thread(step='conversation')

                # Load a random source data to the LLM as a background memory about the context
                source_data = utils.load_one_source_data(source_dir, all_source_files, curr_context)
                if curr_context == 'writing':
                    """
                    Besides other contexts, we introduce the creative writing when evaluating the LLM's ability to generate persona-aligned new contents.
                    It is meaningful as a special case since it is (1) practically useful (2) need to translate writing samples into conversations (3) does not involve personal historical events as in other contexts.
                    """
                    # Convert the writing sample into a conversation
                    preferences = LLM.query_llm(step='prepare_new_content', data=persona, action='preferences', verbose=args['inference']['verbose'])
                    updated_writing_sample = LLM.query_llm(step='prepare_new_content', data=source_data, action='rewrite_from_persona', verbose=args['inference']['verbose'])
                    if 'python' in preferences or 'plaintext' in preferences:
                        preferences = preferences.strip("```python").strip("```plaintext").strip()
                    if 'python' in updated_writing_sample or 'plaintext' in updated_writing_sample:
                        updated_writing_sample = updated_writing_sample.strip("```python").strip("```plaintext").strip()

                    # writing_styles, formatting_styles, updated_writing_sample = utils.rewrite_sample(LLM, persona, source_data, verbose=args['inference']['verbose'])
                    conversation = LLM.query_llm(step='prepare_new_content', action='rewrite_as_conversation', verbose=args['inference']['verbose'])
                    if 'python' in conversation or 'plaintext' in conversation:
                        conversation = ast.literal_eval(conversation.strip("```python").strip("```plaintext").strip())

                    responses = [source_data, preferences, updated_writing_sample, conversation]
                    data_names = ['Original Sample', 'Writing and Formatting Styles', 'Updated Writing Sample', 'Conversation']
                    for response, data_name in zip(responses, data_names):
                        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)

                else:
                    # Feed the thread with a seeding data from the real-world conversation
                    context_conversation = utils.preprocess_source_data(source_data, curr_context)
                    _ = LLM.query_llm(step='source_data', seed=context_conversation, verbose=args['inference']['verbose'])

                    # Generate general and contextual personal histories across time frames
                    steps = ['init_general_personal_history', 'first_expand_general_personal_history', 'second_expand_general_personal_history', 'third_expand_general_personal_history',
                             'init_contextual_personal_history', 'first_expand_contextual_personal_history', 'second_expand_contextual_personal_history', 'third_expand_contextual_personal_history']
                    data_names = ['Init General Personal History', 'General Personal History Next Week', 'General Personal History Next Month', 'General Personal History Next Year',
                                  'Init Contextual Personal History', 'Contextual Personal History Next Week', 'Contextual Personal History Next Month', 'Contextual Personal History Next Year']
                    existing_general_personal_history = {'init_general_personal_history': LLM.init_general_personal_history, 'first_expand_general_personal_history': LLM.first_expand_general_personal_history,
                                                         'second_expand_general_personal_history': LLM.second_expand_general_personal_history, 'third_expand_general_personal_history': LLM.third_expand_general_personal_history}

                    for step, data_name in zip(steps, data_names):
                        # Only generate general personal history once, to be shared across multiple contexts for the same persona
                        if idx_context > 0 and step in existing_general_personal_history:
                            utils.append_json_to_file(existing_general_personal_history[step], output_file_path, curr_data_name=data_name, parse_json=True)
                            continue

                        response = LLM.query_llm(step=step, persona=expanded_persona, context=curr_context, idx_context=idx_context, start_time=start_time, verbose=args['inference']['verbose'])
                        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True)

                    # Populate personal history into conversation
                    steps = ['init_conversation', 'first_expand_conversation', 'second_expand_conversation', 'third_expand_conversation']
                    data_names = ['Init Conversation', 'Conversation Next Week', 'Conversation Next Month', 'Conversation Next Year']

                    for step, data_name in zip(steps, data_names):
                        response = LLM.query_llm(step=step, context=curr_context, idx_context=idx_context, start_time=start_time, verbose=args['inference']['verbose'])
                        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True)
