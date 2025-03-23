import os
import json
import random
import argparse
import yaml
import re
import torch
from tqdm import tqdm
import csv

from openai import OpenAI


class Evaluation:
    def __init__(self, args, cmd_args):
        self.args = args

        # Load API keys
        token_path = cmd_args.token_path
        with open(os.path.join(token_path, "openai_key.txt"), "r") as api_key_file:
            self.openai_key = api_key_file.read()

        self.client = OpenAI(api_key=self.openai_key)

    def query_llm(self, question, all_options, context=None, instructions=None, verbose=False):
        assert context is None or isinstance(context, list), "Context must be a list of dictionaries"
        if instructions is None:
            instructions = "Find the most appropriate model response and give your final answer (a), (b), (c), or (d) after the special token <final_answer>."
        if context:
            messages = context + [{"role": "user", "content": question + '\n\n' + instructions + '\n\n' + all_options},]
        else:
            messages = [{"role": "user", "content": question + '\n\n' + instructions + '\n\n' + all_options},]

        if 'o' in self.args['models']['llm_model']:
            messages = convert_role_system_to_user(messages)

        response = self.client.chat.completions.create(
            model=self.args['models']['llm_model'],
            messages=messages,
        )
        response = response.choices[0].message.content

        if verbose:
            print("model response: ", response)

        return response

    def extract_answer(self, predicted_answer, correct_answer):
        predicted_answer = predicted_answer.strip()
        if "<final_answer>" in predicted_answer:
            predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
        if predicted_answer.endswith("</final_answer>"):
            predicted_answer = predicted_answer[:-len("</final_answer>")].strip()

        # Extract all option letters (with or without parentheses)
        option_pattern = r'\(?([a-zA-Z])\)?'
        predicted_options = re.findall(option_pattern, predicted_answer)
        predicted_options_lower = list(map(str.lower, predicted_options))

        # Find correct answer letter
        correct_letter = re.search(r'\(?([a-zA-Z])\)?', correct_answer).group(1)
        correct_letter_mentioned = correct_letter.lower() in predicted_options_lower

        # Create a list of incorrect options
        incorrect_letters = [chr(i) for i in range(65, 91) if chr(i) != correct_letter.upper()]
        incorrect_letters_mentioned = any(
            letter.lower() in predicted_options_lower for letter in incorrect_letters
        )

        if correct_letter_mentioned and not incorrect_letters_mentioned:
            return True, predicted_answer
        else:
            return False, predicted_answer


def convert_role_system_to_user(messages_4o):
    """
    Convert OpenAI 4o-style messages (with 'system') to 4.0-style (no 'system').
    - System messages are merged into the next message (as a prefix).
    - Consecutive messages with the same role are merged into one.

    Args:
        messages_4o (list of dict): List of messages with roles like 'system', 'user', 'assistant'.

    Returns:
        list of dict: Cleaned message history for models without system role support.
    """
    messages_o1 = []
    system_buffer = ""

    for msg in messages_4o:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_buffer += f"[System]: {content}\n"
            continue

        # Prepend system message if buffered
        if system_buffer:
            content = system_buffer + content
            system_buffer = ""

        # Merge with previous message if role is the same
        if messages_o1 and messages_o1[-1]["role"] == role:
            messages_o1[-1]["content"] += "\n" + content
        else:
            messages_o1.append({"role": role, "content": content})

    return messages_o1


def build_jsonl_index(jsonl_path):
    """
    Scan the JSONL file once to build a mapping: {key: file_offset}.
    Assumes each line is a JSON object with a single key-value pair.
    """
    index = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def load_context_by_id(jsonl_path, offset):
    """
    Seek to a known offset in the JSONL and load exactly that line.
    Returns the value associated with the single key in the JSON object.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


def load_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value
            yield row_data


def load_rows_with_context(csv_path, jsonl_path):
    jsonl_index = build_jsonl_index(jsonl_path)

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        prev_sid = None
        prev_context = None

        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value

            sid = row_data["shared_context_id"]
            if sid != prev_sid:
                current_context = load_context_by_id(jsonl_path, jsonl_index[sid])

                prev_sid = sid
                prev_context = current_context
            else:
                current_context = prev_context

            yield row_data, current_context


def count_csv_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header row


def run_evaluation(args, cmd_args, llm, verbose=False):
    question_path = cmd_args.question_path
    context_path = cmd_args.context_path
    result_path = cmd_args.result_path

    if os.path.exists(result_path):
        os.remove(result_path)

    all_errors = []
    total_rows = count_csv_rows(question_path)

    for row_data, context in tqdm(load_rows_with_context(question_path, context_path), total=total_rows):
        try:
            # Extract relevant data from the row
            persona_id = row_data["persona_id"]
            question_id = row_data["question_id"]
            question_type = row_data["question_type"]
            topic = row_data["topic"]
            stereotypical = row_data["stereotypical"]
            context_length_in_tokens = row_data["context_length_in_tokens"]
            context_length_in_letters = row_data["context_length_in_letters"]
            distance_to_ref_in_blocks = row_data["distance_to_ref_in_blocks"]
            distance_to_ref_in_tokens = row_data["distance_to_ref_in_tokens"]
            num_irrelevant_tokens = row_data["num_irrelevant_tokens"]
            distance_to_ref_proportion_in_context = row_data["distance_to_ref_proportion_in_context"]
            question = row_data["user_question_or_message"]
            correct_answer = row_data["correct_answer"]
            all_options = row_data["all_options"]
            shared_context_id = row_data["shared_context_id"]
            end_index_in_shared_context = row_data["end_index_in_shared_context"]

            # Prepare the context for the LLM query
            context= context[:int(end_index_in_shared_context)]  # Include up to the end index

            # Send the query to the LLM
            model_response = llm.query_llm(question, all_options, context)
            score, predicted_answer = llm.extract_answer(model_response, correct_answer)

            # Save the results back to a CSV file together with the question types
            if verbose:
                print(f"Question: {question}")
                print(f"Predicted Answer: {predicted_answer}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Score: {score}")

            with open(result_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write the header if the file is empty
                if os.stat(result_path).st_size == 0:
                    writer.writerow(["score", "persona_id", "question_id", "user_question_or_message", "question_type", "topic", "stereotypical", "context_length_in_tokens", "context_length_in_letters",
                                     "distance_to_ref_in_blocks", "distance_to_ref_in_tokens", "num_irrelevant_tokens", "distance_to_ref_proportion_in_context",
                                     "model_response", "len_of_model_response", "predicted_answer", "correct_answer"])
                writer.writerow([
                    score,
                    persona_id,
                    question_id,
                    question,
                    question_type,
                    topic,
                    stereotypical,
                    context_length_in_tokens,
                    context_length_in_letters,
                    distance_to_ref_in_blocks,
                    distance_to_ref_in_tokens,
                    num_irrelevant_tokens,
                    distance_to_ref_proportion_in_context,
                    model_response,
                    len(model_response),
                    predicted_answer,
                    correct_answer,
                ])
        except Exception as e:
            print(f"Error: {e}")
            all_errors.append({
                "persona_id": row_data["persona_id"],
                "question_id": row_data["question_id"],
                "error": str(e)
            })
            continue

    if all_errors:
        for error in all_errors:
            print(f"Error for persona_id {error['persona_id']} and question_id {error['question_id']}: {error['error']}")


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')

    """ General arguments """
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from o3-mini, o1, o1-mini, gpt-4o, gpt-4o-mini, '
                                                                    'Llama-3.3-70B-Instruct, Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-8B-Instruct, '
                                                                    'claude-3-7-sonnet-20250219, DeepSeek-R1, DeepSeek-v3,'
                                                                    'gemini-2.0-flash, gemini-1.5-flash, gemini-2.0-flash-lite')
    parser.add_argument('--step', type=str, default='prepare', help='Step to run: prepare or evaluate')
    parser.add_argument('--token_path', type=str, default='api_tokens', help='Path to the API tokens')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing csv and json files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    """ Arguments for running the evaluation step """
    parser.add_argument('--question_path', type=str, default='data/questions_128k.csv', help='Path to the questions CSV file')
    parser.add_argument('--context_path', type=str, default='data/shared_contexts_128k.jsonl', help='Path to the contexts JSONL file')
    parser.add_argument('--result_path', type=str, default='data/eval_results.csv', help='Path to save the results CSV file')

    cmd_args = parser.parse_args()
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']

    base_dir = "./data/output"
    llm = Evaluation(args, cmd_args)

    if cmd_args.step == 'evaluate':
        run_evaluation(args, cmd_args, llm, verbose=cmd_args.verbose)
    else:
        raise ValueError("Invalid step. Choose 'prepare' or 'evaluate'.")
