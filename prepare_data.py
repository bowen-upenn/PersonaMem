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
    # found = False
    if found:
        # Ensure that every data file with the same idx_persona share the same persona
        persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year \
            = found['persona'], found['expanded_persona'], found['start_time'], found['init_general_personal_history'], found['general_personal_history_next_week'], found['general_personal_history_next_month'], found['general_personal_history_next_year']
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
        init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year = None, None, None, None

    return persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year


def prepare_topics(idx_topic, all_topics, curr_topic, args):
    # Process each topic as needed
    print(f'{utils.Colors.OKGREEN}Processing topic: {curr_topic}, {idx_topic + 1}/{len(all_topics)}{utils.Colors.ENDC}')

    # Load a random conversation history from the chosen real-world dataset
    if curr_topic == 'writing':
        source_dir = args['datasets']['writing_source_dir']
    elif curr_topic == 'email':
        source_dir = args['datasets']['email_source_dir']
    elif curr_topic == 'coding':
        source_dir = args['datasets']['code_source_dir']
    elif curr_topic == 'legal':
        source_dir = args['datasets']['legal_source_dir']
    elif curr_topic == 'therapy':
        source_dir = args['datasets']['therapy_source_dir']
    else:
        source_dir = None
        print(f'{utils.Colors.WARNING}No source data is available for the topic: {curr_topic}{utils.Colors.ENDC}')

    all_source_files = utils.load_all_source_data(source_dir, curr_topic) if source_dir is not None else None
    return source_dir, all_source_files


def parse_conversation_sections(LLM, input_conversation, topic, last_timestamp, verbose):
    """
    :param input_conversation: A list of strings representing the conversation
    We define each section in the conversation as a group of lines before the next Side_Note
    """
    def expand_section(LLM, section, last_timestamp):
        if verbose:
            print(f'{utils.Colors.OKGREEN}{"Original Section"}:{utils.Colors.ENDC}')
            print(section)

        response = LLM.query_llm(step='expand_conversation_section', topic=topic, data={'section': section, 'last_timestamp': last_timestamp}, verbose=False)
        match = re.search(r'```(?:python|plaintext)?\s*(.*?)\s*```', response, re.DOTALL)
        response = match.group(1) if match else response
        response = response.strip().replace('\n', '')
        if '=' in response:
            response = re.sub(r'^\s*\w+\s*=\s*', '', response, count=1).strip()
        if response[-1] != ']':
            response += ']'
        if response[-2] != '"' and response[-3] == '"':
            response = response[:-3] + '"]'

        if verbose:
            print(f'{utils.Colors.OKGREEN}{"Expanded Section"}:{utils.Colors.ENDC}')
            print(response)
        # response = response.strip("```python").strip("```plaintext").strip()
        # for parser in ast.literal_eval:
        # try:
        if verbose:
            print('Parsed section', response, '\n\n')
        response = ast.literal_eval(response)
        return response
        # except:
        #     return response
            # continue  # Try the next parser
        # return response

    # Keywords to identify the start of a new section
    keywords = {'Side_Note', 'Side_Notes', '[Side_Note]', '[Side_Notes]', 'Side', '[Side'}
    sections = []  # To store the parsed sections
    with_next_sidenote = []
    current_section = []  # To collect strings for the current section

    # print('input_conversation', input_conversation, '\n\n')
    match = re.search(r'```(?:python|plaintext)?\s*(.*?)\s*```', input_conversation, re.DOTALL)
    input_conversation = match.group(1) if match else input_conversation
    input_conversation = input_conversation.strip().replace('\n', '')
    if '=' in input_conversation:
        input_conversation = re.sub(r'^\s*\w+\s*=\s*', '', input_conversation, count=1).strip()
    if input_conversation[-1] != ']':
        input_conversation += ']'
    if verbose:
        print('parsed input_conversation', input_conversation, '\n\n')
    # input_conversation = input_conversation.strip("```python").strip("```plaintext").strip()
    input_conversation = ast.literal_eval(input_conversation)
    # print('input_conversation', input_conversation, '\n\n')

    for idx, line in enumerate(input_conversation):
        # Check if the line starts with any of the keywords
        if any(line.startswith(keyword) for keyword in keywords):
            # Save the current section (if not empty) and start a new one
            if current_section:
                # # Add the next line containing the next Side_Note, if any, to support smoother transition
                # if idx + 1 < len(input_conversation):
                #     current_section.append(input_conversation[idx + 1])
                sections.append(current_section)
                current_section = []
        # Add the current line to the current section
        current_section.append(line)

    # Add the last section if there is one
    if current_section:
        sections.append(current_section)
    # print('all sections', sections, '\n\n')

    expanded_conversation = []
    for idx, section in enumerate(sections):
        # print('section', section, '\n\n')
        expanded_section = expand_section(LLM, section, last_timestamp)

        if idx == 0:
            if any(expanded_section[0].startswith(keyword) for keyword in keywords) and not any(section[0].startswith(keyword) for keyword in keywords):
                expanded_conversation += expanded_section[1:]  # Remove extra side note not existed in the original data, resulting from the prompt template to expand those sections
        else:
            expanded_conversation += expanded_section

    if verbose:
        print(f'{utils.Colors.OKGREEN}{"Expanded Conversation"}:{utils.Colors.ENDC}')
        print(expanded_conversation)

    return expanded_conversation


def prepare_data_on_writing_topic(LLM, topic, persona, source_data, output_file_path, args):
    # Convert the writing sample into a conversation
    preferences = LLM.query_llm(step='prepare_new_content', data=persona, action='preferences', data_type=topic, verbose=args['inference']['verbose'])
    if topic == 'coding':
        source_data = LLM.query_llm(step='translate_code', persona=preferences, data=source_data, verbose=args['inference']['verbose'])
    elif topic == 'email':
        source_data = LLM.query_llm(step='rewrite_email', persona={'persona': persona, 'preferences': preferences}, data=source_data, verbose=args['inference']['verbose'])
    elif topic == 'writing':
        source_data = LLM.query_llm(step='rewrite_creative_writing', persona={'persona': persona, 'preferences': preferences}, data=source_data, verbose=args['inference']['verbose'])

    updated_writing_sample = LLM.query_llm(step='prepare_new_content', data=source_data, action='rewrite_from_persona', data_type=topic, verbose=args['inference']['verbose'])
    if 'python' in preferences or 'plaintext' in preferences:
        preferences = preferences.strip("```python").strip("```plaintext").strip()
    if 'plaintext' in updated_writing_sample:
        updated_writing_sample = updated_writing_sample.strip("```plaintext").strip()

    conversation = LLM.query_llm(step='prepare_new_content', action='rewrite_as_conversation', data_type=topic, verbose=args['inference']['verbose'])
    if conversation.startswith('```python'):
        conversation = conversation.replace('```python', '', 1)
    conversation = conversation.strip("```plaintext")
    try:
        conversation = json.loads(conversation)
    except:
        conversation = conversation

    # if 'python' in conversation or 'plaintext' in conversation:
    #     conversation = conversation.strip("```plaintext").replace('```python', '', 1).strip()
    #     conversation = ast.literal_eval(conversation)
    # # conversation.append("User: Could you please help me write another sample?")

    responses = [source_data, preferences, updated_writing_sample, conversation]
    if topic == 'coding':
        data_names = ['Original Sample', 'Coding and Formatting Styles', 'Updated Coding Sample', 'Conversation']
    else:
        data_names = ['Original Sample', 'Writing and Formatting Styles', 'Updated Writing Sample', 'Conversation']
    for response, data_name in zip(responses, data_names):
        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)


def prepare_data_on_other_topics(LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
                                 init_general_personal_history, first_expand_general_personal_history, second_expand_general_personal_history, third_expand_general_personal_history, args):
    # Feed the thread with a seeding data from the real-world conversation
    if source_dir is not None:
        source_conversation = utils.preprocess_source_data(source_data, curr_topic)
        _ = LLM.query_llm(step='source_data', seed=source_conversation, verbose=args['inference']['verbose'])
    else:
        _ = LLM.query_llm(step='elaborate_topic', topic=curr_topic, verbose=args['inference']['verbose'])

    # Generate general and contextual personal histories across time frames
    steps = ['init_general_personal_history', 'first_expand_general_personal_history', 'second_expand_general_personal_history', 'third_expand_general_personal_history',
             'init_contextual_personal_history', 'first_expand_contextual_personal_history', 'second_expand_contextual_personal_history', 'third_expand_contextual_personal_history']
    data_names = ['Init General Personal History', 'General Personal History Next Week', 'General Personal History Next Month', 'General Personal History Next Year',
                  'Init Contextual Personal History', 'Contextual Personal History Next Week', 'Contextual Personal History Next Month', 'Contextual Personal History Next Year']
    existing_general_personal_history = {'init_general_personal_history': init_general_personal_history, 'first_expand_general_personal_history': first_expand_general_personal_history,
                                         'second_expand_general_personal_history': second_expand_general_personal_history, 'third_expand_general_personal_history': third_expand_general_personal_history}
    # steps = ['init_general_personal_history', 'init_contextual_personal_history']
    # data_names = ['Init General Personal History', 'Init Contextual Personal History']
    # existing_general_personal_history = {'init_general_personal_history': LLM.init_general_personal_history}

    last_timestamps = []
    for step, data_name in tqdm(zip(steps, data_names)):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')
        # Only generate general personal history once, to be shared across multiple topics for the same persona
        # if idx_topic > 0 and step in existing_general_personal_history:
        #     utils.append_json_to_file(existing_general_personal_history[step], output_file_path, curr_data_name=data_name, parse_json=True)
        #     continue
        if step in existing_general_personal_history:
            if existing_general_personal_history[step] is not None:
                # Use existing general personal history shared across multiple topics for the same persona
                utils.append_json_to_file('```json' + str(existing_general_personal_history[step]) + '```', output_file_path, curr_data_name=data_name, parse_json=True)
                continue

        response = LLM.query_llm(step=step, persona=expanded_persona, topic=curr_topic, idx_topic=idx_topic, start_time=start_time, verbose=args['inference']['verbose'])
        utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=True)
        last_timestamps.append(utils.extract_last_timestamp(response))

    # Populate personal history into conversation
    steps = ['init_conversation', 'first_expand_conversation', 'second_expand_conversation', 'third_expand_conversation']
    data_names = ['Init Conversation', 'Conversation Next Week', 'Conversation Next Month', 'Conversation Next Year']
    # steps = ['init_conversation']
    # data_names = ['Init Conversation']

    last_timestamps = utils.merge_timestamps(last_timestamps)
    for conv_idx, (step, data_name) in enumerate(zip(steps, data_names)):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')
        response = LLM.query_llm(step=step, topic=curr_topic, idx_topic=idx_topic, start_time=start_time, verbose=args['inference']['verbose'])
        response = LLM.query_llm(step='reflect_' + step, topic=curr_topic, data=response, action=1, verbose=args['inference']['verbose'])
        response = LLM.query_llm(step='reflect_' + step, topic=curr_topic, action=2, verbose=args['inference']['verbose'])
        expanded_conversation = parse_conversation_sections(LLM, response, curr_topic, last_timestamps[conv_idx], verbose=args['inference']['verbose'])
        utils.append_json_to_file(expanded_conversation, output_file_path, curr_data_name=data_name, parse_json=False, parse_list=False)


def prepare_irrelevant_contexts(LLM, args):
    with open(args['datasets']['random_questions_file'], 'r') as file:
        all_random_questions = [line.strip() for line in file]
    with open(args['datasets']['random_code_questions_file'], 'r') as file:
        all_random_code_questions = [line.strip() for line in file]
    all_random_questions = all_random_questions + all_random_code_questions

    output_file_path = 'data/irrelevant_contexts.json'
    for index, question in enumerate(tqdm(all_random_questions)):
        LLM.create_a_thread(step='irrelevant')

        model_answer = LLM.query_llm(step='random_question', data=question, verbose=args['inference']['verbose'])
        follow_up_question = LLM.query_llm(step='random_question_follow_up', verbose=args['inference']['verbose'])
        follow_up_answer = LLM.query_llm(step='random_question_follow_up_response', data=follow_up_question, verbose=args['inference']['verbose'])

        new_entry = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": model_answer},
            {"role": "user", "content": follow_up_question},
            {"role": "assistant", "content": follow_up_answer}
        ]

        LLM.delete_a_thread(step='irrelevant')

        existing_data = []
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except json.JSONDecodeError:
                existing_data = []

        existing_data.append({str(index): new_entry})
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4)


def prepare_data(args):
    # Load all personas
    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()
    LLM = QueryLLM(args)

    if args['datasets']['topics'] == ['irrelevant']:
        prepare_irrelevant_contexts(LLM, args)
    else:
        # Generate conversational data relevant to the topic and the persona
        all_errored_data_paths = {}

        for idx_persona in tqdm(range(int(args['inference']['start_persona_idx']), int(args['inference']['num_personas']))):
            persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, \
                general_personal_history_next_month, general_personal_history_next_year = prepare_persona(LLM, idx_persona, all_personas, args)

            # Clean up the names of topics
            if args['datasets']['topics'] == ['all']:
                all_topics = utils.get_all_topic_names()
            else:
                all_topics = [ctx.strip() for ctx in args['datasets']['topics']]

            # Since we assign a consecutive time frame for all topics, we randomly permute topics to ensure generalization
            if len(all_topics) > 1:
                random.shuffle(all_topics)

            # Loop through each topic in the list
            for idx_topic, curr_topic in tqdm(enumerate(all_topics)):
                source_dir, all_source_files = prepare_topics(idx_topic, all_topics, curr_topic, args)

                # Set a consecutive time frame for different topics for each persona, while all samples below are independent
                if idx_topic > 0:
                    start_time = utils.pick_a_random_time_within_a_year(start_time)

                for idx_sample in range(int(args['inference']['start_sample_idx']), int(args['inference']['num_samples_per_topic'])):
                    output_file_path = os.path.join(args['inference']['output_dir'],
                                                    os.path.join(f'{curr_topic}', f'{args["inference"]["output_file_name"]}_{curr_topic}_persona{idx_persona}_sample{idx_sample}.json'))
                    utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                    utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                    utils.append_json_to_file(curr_topic, output_file_path, curr_data_name='Topic', parse_json=False)
                    print(f'{utils.Colors.OKGREEN}Output file path: {output_file_path}{utils.Colors.ENDC}')

                    LLM.create_a_thread(step='conversation')

                    # Load a random source data to the LLM as a background memory about the topic
                    source_data = utils.load_one_source_data(source_dir, all_source_files, curr_topic) if all_source_files is not None else None
                    try:
                        if curr_topic == 'writing' or curr_topic == 'email' or curr_topic == 'coding':
                            """
                            Besides other topics, we introduce the creative writing, email writing, and code programming when evaluating the LLM's ability to generate persona-aligned new contents.
                            It is meaningful as a special case since it is (1) practically useful (2) need to translate writing samples into conversations (3) does not involve personal historical events as in other topics.
                            """
                            prepare_data_on_writing_topic(LLM, curr_topic, persona, source_data, output_file_path, args)
                        else:
                            prepare_data_on_other_topics(LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
                                                         init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year, args)
                    except Exception as e:
                        print(f'{utils.Colors.FAIL}Error at generating file{output_file_path}: {e}{utils.Colors.ENDC}')
                        all_errored_data_paths[output_file_path] = e

                    LLM.delete_a_thread(step='conversation')

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
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['datasets']['topics'] = cmd_args.topics if cmd_args.topics is not None else args['datasets']['topics']
    args['inference']['num_personas'] = cmd_args.n_persona if cmd_args.n_persona is not None else args['inference']['num_personas']
    args['inference']['num_samples_per_topic'] = cmd_args.n_samples if cmd_args.n_samples is not None else args['inference']['num_samples_per_topic']
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
