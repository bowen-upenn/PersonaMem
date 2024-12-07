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

    for idx_persona in range(int(args['inference']['num_personas'])):
        # Load a random persona
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

            for idx_sample in range(int(args['inference']['num_samples_per_context'])):
                output_file_path = os.path.join(args['inference']['output_dir'], f'{args["inference"]["output_file_name"]}_{curr_context}_persona{idx_persona}_sample{idx_sample}.json')
                utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                utils.append_json_to_file(curr_context, output_file_path, curr_data_name='Context', parse_json=False)

                LLM.create_a_thread()

                # Load a random source data to the LLM as a background memory about the context
                source_data = utils.load_one_source_data(all_source_files, curr_context)
                if curr_context == 'writing':
                    """
                    Besides other contexts, we introduce the creative writing when evaluating the LLM's ability to generate persona-aligned new contents.
                    It is meaningful as a special case since it is (1) practically useful (2) need to translate writing samples into conversations (3) does not involve personal historical events as in other contexts.
                    """
                    # Convert the writing sample into a conversation
                    writing_styles, formatting_styles, updated_writing_sample = utils.rewrite_sample(LLM, persona, source_data, verbose=args['inference']['verbose'])
                    conversation = LLM.query_llm(step='new_content', action='rewrite_as_conversation', verbose=args['inference']['verbose'])
                    new_writing_sample = LLM.query_llm(step='new_content', data={'persona': persona, 'writing_styles': writing_styles, 'formatting_styles': formatting_styles}, action='write_new_sample', verbose=False)

                    utils.append_json_to_file(writing_styles, output_file_path, curr_data_name='Writing Styles', parse_json=False)
                    utils.append_json_to_file(formatting_styles, output_file_path, curr_data_name='Formatting Styles', parse_json=False)
                    utils.append_json_to_file(updated_writing_sample, output_file_path, curr_data_name='Updated Writing Sample', parse_json=False)
                    utils.append_json_to_file(conversation, output_file_path, curr_data_name='Conversation', parse_json=False)
                    utils.append_json_to_file(new_writing_sample, output_file_path, curr_data_name='New Writing Sample', parse_json=False)

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
