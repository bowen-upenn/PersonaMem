import os
import json
import random
import tiktoken
import argparse
import yaml
import re
import torch
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

import utils
import prepare_blocks

# OpenAI ChatGPT API
from openai import OpenAI

# Google Gemini API from VertexAI
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession

# Meta Llama API from Replicate
import replicate

# Anthropic Claude API
import anthropic


def evaluate_answer(predicted_answer, correct_answer):
    """
    Evaluate the answer based on two criteria:
    1. If the correct option (e.g., (a)) is mentioned, and incorrect options are not.
    2. Otherwise, use SentenceBERT to check similarity.
    """
    # Extract all option patterns (e.g., (a), (b), etc.) from the answer
    option_pattern = r'\(([a-zA-Z])\)'
    predicted_options = re.findall(option_pattern, predicted_answer)

    # Check for the correct and incorrect options
    correct_letter_mentioned = correct_answer.lower() in map(str.lower, predicted_options)
    incorrect_letters = [chr(i) for i in range(65, 91) if chr(i) != correct_answer.upper()]
    incorrect_letters_mentioned = any(
        letter.lower() in map(str.lower, predicted_options) for letter in incorrect_letters
    )

    if correct_letter_mentioned and not incorrect_letters_mentioned:
        return True  # Match based on Criterion 1

    # Criterion 2: Use SentenceBERT similarity
    similarity = util.pytorch_cos_sim(
        sentence_bert_model.encode(correct_answer, convert_to_tensor=True),
        sentence_bert_model.encode(predicted_answer, convert_to_tensor=True)
    ).item()

    return similarity > 0.8  # Match if similarity > 0.8


class Evaluation:
    def __init__(self, args):
        self.args = args

        # Load API keys or tokens
        with open("api_tokens/openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()
        if re.search(r'gemini', self.args['models']['llm_model']) is not None:
            with open("api_tokens/gemini_project_id.txt", "r") as vertexai_project_id_file:
                self.project_id = vertexai_project_id_file.read()
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.args['models']['gemini_credential_path']
        elif re.search(r'llama', self.args['models']['llm_model']) is not None:
            with open("api_tokens/llama_key.txt", "r") as llama_key_file:
                llama_key = llama_key_file.read()
            os.environ['REPLICATE_API_TOKEN'] = llama_key
        elif re.search(r'claude', self.args['models']['llm_model']) is not None:
            with open("api_tokens/claude_key.txt", "r") as claude_key_file:
                self.claude_key = claude_key_file.read()
        elif re.search(r'mistral', self.args['models']['llm_model']) is not None:
            with open("api_tokens/mistral_key.txt", "r") as mistral_key_file:
                self.mistral_key = mistral_key_file.read()


    def query_llm(self, all_conversations, formatted_question, which_format):
        if which_format == 'string':
            messages = [{"role": "user", "content": all_conversations + '\n\n' + formatted_question}],
        elif which_format == 'api_dict':
            messages = all_conversations + [{"role": "user", "content": formatted_question}]
        else:
            raise ValueError(f"Format {which_format} is not supported.")

        # Call OpenAI API for GPT models by default
        if re.search(r'gpt', self.args['models']['llm_model']) is not None or re.search(r'o1', self.args['models']['llm_model']) is None:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.args['models']['llm_model'],
                messages=messages,
            )
            response = response.choices[0].message.content

        # Call Google Gemini API for Gemini models
        elif re.search(r'gemini', self.args['models']['llm_model']) is not None:
            location = "us-central1"
            vertexai.init(project=self.project_id, location=location)
            model = GenerativeModel(self.args['models']['llm_model'])

            prompt = ' '.join(msg['content'] for msg in messages)
            if re.search(r'1.5', self.args['models']['llm_model']) is not None:  # it supports vision though it is not used in this project
                response = model.generate_content([prompt]).text
            else:
                chat = model.start_chat()
                response = chat.send_message(prompt).text

        # Call Meta Llama API for Llama models
        elif re.search(r'llama', self.args['models']['llm_model']) is not None:
            prompt = ' '.join(msg['content'] for msg in messages)
            response = ""
            try:
                for event in replicate.stream(
                        "meta/" + self.args['models']['llm_model'],
                        input={
                            "prompt": prompt,
                        },
                ):
                    response += str(event)
            except Exception as e:
                response = ""

        # Call Anthropic Claude API for Claude models
        elif re.search(r'claude', self.args['models']['llm_model']) is not None:
            client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=self.claude_key,
            )
            response = client.messages.create(
                model=self.args['models']['llm_model'],
                system=messages[0]['content'],
                messages=messages[1:]
            ).content[0].text

        else:
            raise ValueError(f"Model {self.args['models']['llm_model']} is not supported.")

        return response


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
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt4", help='Set LLM model. Choose from o1-preview, o1-mini, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, '
                                                                  'gemini-1.5-flash-002, gemini-1.5-pro-002, gemini-1.0-pro'
                                                                  'meta-llama-3-70b-instruct, meta-llama-3-8b-instruct,'
                                                                  'claude-3-opus-20241022, claude-3-5-sonnet-20241022, claude-3-sonnet-20241022')
    parser.add_argument('--idx_persona', type=int, default=0, help='Index of the persona')
    parser.add_argument('--format', type=str, default='string', help='Output conversation format: string or api_dict. Not applicable for qa')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of conversation blocks')
    parser.add_argument('--up_to', dest='accumulate', action='store_true', help='Generate up-to n_blocks, not just n_blocks itself')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    cmd_args = parser.parse_args()
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']

    llm_model = cmd_args.model
    idx_persona = cmd_args.idx_persona

    which_format = cmd_args.format
    verbose = cmd_args.verbose

    base_dir = "./data/output"
    evaluation = Evaluation(args)
    tokenizer = tiktoken.encoding_for_model(args['models']['llm_model'])
    sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    if cmd_args.up_to is False:
        n_blocks = [cmd_args.n_blocks]
    else:
        n_blocks = range(1, cmd_args.n_blocks)

    for curr_n_blocks in n_blocks:
        output_file_path = f'./data/eval/{llm_model}_{idx_persona}_{curr_n_blocks}.json'
        results = {}

        # Gather all candidate conversation blocks
        chosen_blocks = load_n_conversation_blocks(idx_persona, curr_n_blocks, base_dir, verbose)

        # Process each chosen conversation block
        processed_blocks_dict = {}
        all_strings = []

        for (file_name, time_period), conversation in chosen_blocks:
            context = file_name.split('_')[1]
            processed_conversation, latest_ts = process_conversation_block(context, conversation, which_format)

            qa = extract_qa(base_dir, context, file_name, time_period)

            processed_blocks_dict[latest_ts] = {
                "conversation": processed_conversation[0],  # idx 0 corresponds to the conversation in the required format, either string or api_dict
                "file_name": file_name,
                "time_period": time_period,
                "last_timestamp": latest_ts,
                "context": context,
                "qa": qa
            }
            all_strings.append(processed_conversation[-1])  # idx -1 always corresponds to the conversation in the plain string format

        # Topological sort chosen conversation blocks by the latest timestamp
        sorted_processed_blocks = topological_sort(processed_blocks_dict, verbose)
        all_qa = compute_question_distance(sorted_processed_blocks)

        # Concatenate all conversation blocks
        all_conversations = concatenate_blocks(sorted_processed_blocks, which_format, verbose)
        count_tokens(all_strings)

        # Show all Q&As related to this concatenated conversation
        for formatted_question, correct_answer, distance, question_type in tqdm(question_loader(all_qa)):
            """
            Example usage: formatted_question -> LLM -> predicted_answer <-> correct_answer
            """
            predicted_answer = evaluation.query_llm(all_conversations, formatted_question, which_format)
            match = evaluate_answer(predicted_answer, correct_answer)

            if verbose:
                print(f'{utils.Colors.OKGREEN}{"Correct answer"}:{utils.Colors.ENDC}{correct_answer}')
                print(f'{utils.Colors.OKGREEN}{"Predicted answer"}:{utils.Colors.ENDC}{predicted_answer}')
                if match:
                    print(f'{utils.Colors.OKGREEN}{"Correct"}:{utils.Colors.ENDC}')
                else:
                    print(f'{utils.Colors.FAIL}{"Incorrect"}:{utils.Colors.ENDC}')

            # Save results based on the distances from the question being asked at the end to the sourced conversation block
            if distance not in results:
                results[distance] = {"correct": 0, "total": 0}
            else:
                results[distance]["total"] += 1
                if match:
                    results[distance]["correct"] += 1

            # Save results based on the question types
            if question_type not in results:
                results[question_type] = {"correct": 0, "total": 0}
            else:
                results[question_type]["total"] += 1
                if match:
                    results[question_type]["correct"] += 1

        # Save evaluation results to a JSON file.
        with open(output_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
