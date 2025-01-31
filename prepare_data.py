from tqdm import tqdm
import os
import numpy as np
import re
import json
import random
import torch
import ast
import yaml
import argparse
import sys
import ast

from query_llm import QueryLLM
import utils


def prepare_persona(LLM, idx_persona, all_personas, args):
    # Load a random persona
    found = utils.find_existing_persona_files(idx_persona)
    if found:
        # Ensure that every data file with the same idx_persona share the same persona
        persona, expanded_persona, start_time = found['persona'], found['expanded_persona'], found['start_time']
        LLM.expanded_persona = expanded_persona
        if not start_time:
            start_time = utils.pick_a_random_time()
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

    return persona, expanded_persona, start_time


def prepare_context(idx_context, all_contexts, curr_context, args):
    # Process each context as needed
    print(f'{utils.Colors.OKGREEN}Processing context: {curr_context}, {idx_context + 1}/{len(all_contexts)}{utils.Colors.ENDC}')

    # Load a random conversation history from the chosen real-world dataset
    if curr_context == 'writing':
        source_dir = args['datasets']['writing_source_dir']
    elif curr_context == 'legal':
        source_dir = args['datasets']['legal_source_dir']
    elif curr_context == 'therapy':
        source_dir = args['datasets']['therapy_source_dir']
    else:
        source_dir = None
        print(f'{utils.Colors.WARNING}No source data is available for the context: {curr_context}{utils.Colors.ENDC}')

    all_source_files = None
    if source_dir is not None:
        all_source_files = utils.load_all_source_data(source_dir, curr_context)

    return source_dir, all_source_files


def parse_conversation_sections(LLM, input_conversation, context, last_timestamp, verbose):
    """
    :param input_conversation: A list of strings representing the conversation
    We define each section in the conversation as a group of lines before the next Side_Note
    """
    def expand_section(LLM, section, last_timestamps):
        response = LLM.query_llm(step='expand_conversation_section', context=context, data={'section': section, 'last_timestamp': last_timestamp}, verbose=verbose)
        response = response.strip("```python").strip("```plaintext").strip()
        response = ast.literal_eval(response)
        return response

    # Keywords to identify the start of a new section
    keywords = {'Side_Note', 'Side_Notes', '[Side_Note]', '[Side_Notes]', 'Side', '[Side'}
    sections = []  # To store the parsed sections
    with_next_sidenote = []
    current_section = []  # To collect strings for the current section

    for idx, line in enumerate(input_conversation):
        # Check if the line starts with any of the keywords
        if any(line.startswith(keyword) for keyword in keywords):
            # Save the current section (if not empty) and start a new one
            if current_section:
                # Add the next line containing the next Side_Note, if any, to support smoother transition
                if idx + 1 < len(input_conversation):
                    current_section.append(input_conversation[idx + 1])
                sections.append(current_section)
                current_section = []
        # Add the current line to the current section
        current_section.append(line)

    # Add the last section if there is one
    if current_section:
        sections.append(current_section)

    expanded_conversation = []
    for idx, section in enumerate(sections):
        expanded_section = expand_section(LLM, section, last_timestamp)
        if idx + 1 < len(sections):
            expanded_conversation += expanded_section[:-1]  # Do not repetitively add the last line of each section, i.e., the Side_Note in the next section
        else:
            expanded_conversation += expanded_section

    return expanded_conversation


def prepare_data_on_writing_context(LLM, persona, source_data, output_file_path, args):
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
    conversation.append("User: Could you please help me write another sample?")

    responses = [source_data, preferences, updated_writing_sample, conversation]
    data_names = ['Original Sample', 'Writing and Formatting Styles', 'Updated Writing Sample', 'Conversation']
    for response, data_name in zip(responses, data_names):
        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)


def prepare_data_on_other_contexts(LLM, expanded_persona, source_data, source_dir, curr_context, idx_context, start_time, output_file_path, args):
    # Feed the thread with a seeding data from the real-world conversation
    if source_dir is not None:
        context_conversation = utils.preprocess_source_data(source_data, curr_context)
        _ = LLM.query_llm(step='source_data', seed=context_conversation, verbose=args['inference']['verbose'])

    # Generate general and contextual personal histories across time frames
    steps = ['init_general_personal_history', 'first_expand_general_personal_history', 'second_expand_general_personal_history', 'third_expand_general_personal_history',
             'init_contextual_personal_history', 'first_expand_contextual_personal_history', 'second_expand_contextual_personal_history', 'third_expand_contextual_personal_history']
    data_names = ['Init General Personal History', 'General Personal History Next Week', 'General Personal History Next Month', 'General Personal History Next Year',
                  'Init Contextual Personal History', 'Contextual Personal History Next Week', 'Contextual Personal History Next Month', 'Contextual Personal History Next Year']
    existing_general_personal_history = {'init_general_personal_history': LLM.init_general_personal_history, 'first_expand_general_personal_history': LLM.first_expand_general_personal_history,
                                         'second_expand_general_personal_history': LLM.second_expand_general_personal_history,
                                         'third_expand_general_personal_history': LLM.third_expand_general_personal_history}
    # steps = ['init_general_personal_history', 'init_contextual_personal_history']
    # data_names = ['Init General Personal History', 'Init Contextual Personal History']
    # existing_general_personal_history = {'init_general_personal_history': LLM.init_general_personal_history}

    last_timestamps = []
    for step, data_name in tqdm(zip(steps, data_names)):
        # Only generate general personal history once, to be shared across multiple contexts for the same persona
        if idx_context > 0 and step in existing_general_personal_history:
            utils.append_json_to_file(existing_general_personal_history[step], output_file_path, curr_data_name=data_name, parse_json=True)
            continue

        response = LLM.query_llm(step=step, persona=expanded_persona, context=curr_context, idx_context=idx_context, start_time=start_time, verbose=args['inference']['verbose'])
        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True)
        last_timestamps.append(utils.extract_last_timestamp(response))

    # Populate personal history into conversation
    steps = ['init_conversation', 'first_expand_conversation', 'second_expand_conversation', 'third_expand_conversation']
    data_names = ['Init Conversation', 'Conversation Next Week', 'Conversation Next Month', 'Conversation Next Year']
    # steps = ['init_conversation']
    # data_names = ['Init Conversation']

    last_timestamps = utils.merge_timestamps(last_timestamps)
    for conv_idx, (step, data_name) in enumerate(zip(steps, data_names)):
        response = LLM.query_llm(step=step, context=curr_context, idx_context=idx_context, start_time=start_time, verbose=args['inference']['verbose'])
        response = LLM.query_llm(step='reflect_' + step, context=curr_context, data=response, action=1, verbose=args['inference']['verbose'])
        response = LLM.query_llm(step='reflect_' + step, context=curr_context, action=2, verbose=args['inference']['verbose'])
        expanded_conversation = parse_conversation_sections(LLM, response, curr_context, last_timestamps[conv_idx], verbose=args['inference']['verbose'])
        utils.append_json_to_file(expanded_conversation, output_file_path, curr_data_name=data_name, parse_json=False, parse_list=False)


def prepare_data(args):
    # Load all personas
    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()
    LLM = QueryLLM(args)

    all_errored_data_paths = {}

    for idx_persona in tqdm(range(int(args['inference']['start_persona_idx']), int(args['inference']['num_personas']))):
        persona, expanded_persona, start_time = prepare_persona(LLM, idx_persona, all_personas, args)

        # Clean up the names of contexts
        if args['datasets']['context'] == ['all']:
            all_contexts = utils.get_all_context_names()
        else:
            all_contexts = [ctx.strip() for ctx in args['datasets']['context']]

        # Since we assign a consecutive time frame for all contexts, we randomly permute contexts to ensure generalization
        if len(all_contexts) > 1:
            random.shuffle(all_contexts)

        # Loop through each context in the list
        for idx_context, curr_context in tqdm(enumerate(all_contexts)):
            source_dir, all_source_files = prepare_context(idx_context, all_contexts, curr_context, args)

            # Set a consecutive time frame for different contexts for each persona, while all samples below are independent
            if idx_context > 0:
                start_time = utils.pick_a_random_time_within_a_year(start_time)

            for idx_sample in range(int(args['inference']['start_sample_idx']), int(args['inference']['num_samples_per_context'])):
                output_file_path = os.path.join(args['inference']['output_dir'],
                                                os.path.join(f'{curr_context}', f'{args["inference"]["output_file_name"]}_{curr_context}_persona{idx_persona}_sample{idx_sample}.json'))
                utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                utils.append_json_to_file(curr_context, output_file_path, curr_data_name='Context', parse_json=False)
                print(f'{utils.Colors.OKGREEN}Output file path: {output_file_path}{utils.Colors.ENDC}')

                LLM.create_a_thread(step='conversation')

                # Load a random source data to the LLM as a background memory about the context
                source_data = None
                if source_dir is not None:
                    source_data = utils.load_one_source_data(source_dir, all_source_files, curr_context)

                try:
                    if curr_context == 'writing':
                        """
                        Besides other contexts, we introduce the creative writing when evaluating the LLM's ability to generate persona-aligned new contents.
                        It is meaningful as a special case since it is (1) practically useful (2) need to translate writing samples into conversations (3) does not involve personal historical events as in other contexts.
                        """
                        prepare_data_on_writing_context(LLM, persona, source_data, output_file_path, args)
                    else:
                        prepare_data_on_other_contexts(LLM, expanded_persona, source_data, source_dir, curr_context, idx_context, start_time, output_file_path, args)
                except Exception as e:
                    print(f'{utils.Colors.FAIL}Error at generating file{output_file_path}: {e}{utils.Colors.ENDC}')
                    all_errored_data_paths[output_file_path] = e
        
    if len(all_errored_data_paths) > 0:
        print(f'{utils.Colors.FAIL}All errored data paths: {utils.Colors.ENDC}')
        for key, value in all_errored_data_paths.items():
            print(key)
    else:
        print(f'{utils.Colors.OKGREEN}All data are successfully generated.{utils.Colors.ENDC}')


if __name__ == "__main__":
    print("Python", sys.version, 'Torch', torch.__version__)
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
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--context', type=str, default="therapy", nargs="+", help='Set conversation context. Choose from therapy, legal, datingConsultation, foodRecommendation, onlineShopping, studyConsultation, travelPlanning, writing'
                                                                                  'or all to select all existing contexts under ./data/output/. '
                                                                                  'If you want to select multiple contexts manually, separate the names by space, e.g. --context therapy legal')  # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument('--n_persona', type=int, default=1, help='Set number of personas to generate')
    parser.add_argument('--n_samples', type=int, default=1, help='Set number of samples per context to generate')
    parser.add_argument('--s_persona', type=int, default=0, help='Set the starting idx of personas to generate')
    parser.add_argument('--s_samples', type=int, default=0, help='Set the starting idx of samples per context to generate')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['datasets']['context'] = cmd_args.context if cmd_args.context is not None else args['datasets']['context']
    args['inference']['num_personas'] = cmd_args.n_persona if cmd_args.n_persona is not None else args['inference']['num_personas']
    args['inference']['num_samples_per_context'] = cmd_args.n_samples if cmd_args.n_samples is not None else args['inference']['num_samples_per_context']
    args['inference']['start_persona_idx'] = cmd_args.s_persona if cmd_args.s_persona is not None else args['inference']['start_persona_idx']
    args['inference']['start_sample_idx'] = cmd_args.s_samples if cmd_args.s_samples is not None else args['inference']['start_sample_idx']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    # Start inference
    print(args)
    if cmd_args.clean:
        user_input = input("The 'clean' flag is set. Do you really want clean up all existing data under ./data/output/? (y/n): ").strip().lower()
        if user_input == 'y':
            utils.clean_up_subdirectories()
        else:
            print("Skipping cleanup.")

    prepare_data(args)
