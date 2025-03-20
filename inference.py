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
import hashlib
import csv
import uuid

import utils
from prepare_blocks import *

# OpenAI ChatGPT API
from openai import OpenAI

# Google Gemini API from VertexAI
# from google import genai
import google.generativeai as genai
# import vertexai
# from vertexai.preview.generative_models import GenerativeModel, ChatSession

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
    # trivial case
    if predicted_answer == correct_answer:
        return True
    
    # Extract all option patterns (e.g., (a), (b), etc.) from the answer
    option_pattern = r'\(([a-zA-Z])\)'
    predicted_options = re.findall(option_pattern, predicted_answer)
    predicted_options_lower = list(map(str.lower, predicted_options))

    # Check for the correct and incorrect options
    correct_letter = re.search(option_pattern, correct_answer).group(1)
    correct_letter_mentioned = correct_letter.lower() in predicted_options_lower
    incorrect_letters = [chr(i) for i in range(65, 91) if chr(i) != correct_letter.upper()]
    incorrect_letters_mentioned = any(
        letter.lower() in predicted_options_lower for letter in incorrect_letters
    )

    if correct_letter_mentioned and not incorrect_letters_mentioned:
        return True  # Match based on Criterion 1

    if len(predicted_answer) == len(correct_answer) == 3:
        # for short answers, don't use SentenceBERT
        return False

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
            self.openai_key = api_key_file.read()
        if re.search(r'gemini', self.args['models']['llm_model']) is not None:
            with open("api_tokens/gemini_key.txt", "r") as gemini_key_file:
                self.gemini_key = gemini_key_file.read()
            # with open("api_tokens/gemini_project_id.txt", "r") as vertexai_project_id_file:
            #     self.project_id = vertexai_project_id_file.read()
            # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.args['models']['gemini_credential_path']
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
        if re.search(r'gpt', self.args['models']['llm_model']) is not None or re.search(r'o1', self.args['models']['llm_model']) is not None:
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model=self.args['models']['llm_model'],
                messages=messages,
            )
            response = response.choices[0].message.content

        # Call Google Gemini API for Gemini models
        elif re.search(r'gemini', self.args['models']['llm_model']) is not None:
            client = genai.Client(api_key=self.gemini_key)
            model = self.args['models']['llm_model']
            response = client.models.generate_content(
                model=model, contents="\n\n".join(msg['content'] for msg in messages)
            )
            response = response.text
            # location = "us-central1"
            # vertexai.init(project=self.project_id, location=location)
            # model = GenerativeModel(self.args['models']['llm_model'])
            #
            # prompt = ' '.join(msg['content'] for msg in messages)
            # if re.search(r'1.5', self.args['models']['llm_model']) is not None:  # it supports vision though it is not used in this project
            #     response = model.generate_content([prompt]).text
            # else:
            #     chat = model.start_chat()
            #     response = chat.send_message(prompt).text

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
                messages=messages[1:],
                max_tokens=1024
            ).content[0].text

        else:
            raise ValueError(f"Model {self.args['models']['llm_model']} is not supported.")

        return response


def generate_conversation_id(context):
    """Generates a unique, fixed-length conversation ID using a hash."""
    return hashlib.sha256(context.encode('utf-8')).hexdigest()[:16]  # First 16 characters of SHA-256 hash


def generate_shared_context_id(shared_context):
    """Returns a consistent ID for the same shared context using hashing."""
    if isinstance(shared_context, list):  # If it's a list, convert it to a string
        shared_context = " ".join(map(str, shared_context))  # Join elements with space

    return hashlib.sha256(shared_context.encode()).hexdigest()  # Generate hash


def save_questions_to_csv(result, csv_file_path="data/questions.csv"):
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header if the file is empty
        if os.stat(csv_file_path).st_size == 0:
            writer.writerow(["persona_id", "question_id", "question_type", "topic", "stereotypical", "context_length_in_tokens", "context_length_in_letters",
                             "distance_to_ref_in_blocks", "distance_to_ref_in_tokens", "num_irrelevant_tokens", "distance_to_ref_proportion_in_context",
                             "question", "correct_answer", "all_options", "shared_context_id", "end_index_in_shared_context"])

        writer.writerow([
            result["idx_persona"],
            result["question_id"],
            result["question_type"],
            result['topic'],
            result["stereotypical"],
            result['context_length_in_tokens'],
            result['context_length_in_letters'],
            result['distance_blocks'],
            result['distance_tokens'],
            result["num_irrelevant_tokens"],
            f"{(result['distance_tokens'] / result['context_length_in_tokens']) * 100:.2f}%",
            result["question"],
            result["correct_answer"],
            result['all_options'],
            result["shared_context_id"],
            result["end_index_in_shared_context"],
        ])


def save_contexts_to_json(contexts_dict, json_file_path="data/contexts.jsonl"):
    """Appends JSON objects to a JSON Lines (NDJSON) file without loading the entire file."""
    with open(json_file_path, "a", encoding="utf-8") as file:  # 'a' mode for append
        file.write(json.dumps(contexts_dict, ensure_ascii=False) + "\n")  # Append as JSONL


def read_jsonl_file(json_file_path="data/contexts.jsonl"):
    """Reads a JSON Lines file line by line into a list of dictionaries."""
    data = []
    with open(json_file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Convert each line to a dictionary
    return data


if __name__ == "__main__":
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    torch.manual_seed(0)
    world_size = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if world_size > 1:
        assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from o1-preview, o1-mini, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, '
                                                                    'gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro'
                                                                    'meta-llama-3-70b-instruct, meta-llama-3-8b-instruct,'
                                                                    'claude-3-opus-20241022, claude-3-5-sonnet-20241022, claude-3-sonnet-20241022')
    parser.add_argument('--idx_persona', type=int, default=0, help='Index of the persona')
    parser.add_argument('--format', type=str, default='api_dict', help='Output conversation format: string or api_dict. Not applicable for qa')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of conversation blocks')
    parser.add_argument('--n_variants', type=int, default=1, help='Number of variants of topological sorts to concatenate conversation sessions')
    parser.add_argument('--up_to', dest='up_to', action='store_true', help='Generate up-to n_blocks, not just n_blocks itself')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing csv and json files and start clean')
    parser.add_argument('--save_only', dest='save_only', action='store_true', help='Save the data only, not evaluating')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    cmd_args = parser.parse_args()
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    benchmark_size = '128k' if cmd_args.n_blocks == 20 else ('1M' if cmd_args.n_blocks == 60 else str(cmd_args.n_blocks) + 'blocks')

    if cmd_args.clean:
        if os.path.exists(f"data/questions_{benchmark_size}.csv"):
            os.remove(f"data/questions_{benchmark_size}.csv")
        if os.path.exists(f"data/contexts_{benchmark_size}.jsonl"):
            os.remove(f"data/contexts_{benchmark_size}.jsonl")
        if os.path.exists(f"data/shared_contexts_{benchmark_size}.jsonl"):
            os.remove(f"data/shared_contexts_{benchmark_size}.jsonl")
    #     user_input = input("The 'clean' flag is set. Do you really want remove existing questions.csv and contexts.json? (y/n): ").strip().lower()
    #     if user_input == 'y':
    #         if os.path.exists("data/questions.csv"):
    #             os.remove("data/questions.csv")
    #         if os.path.exists("data/contexts.json"):
    #             os.remove("data/contexts.json")
    #     else:
    #         print("Skipping cleanup.")

    llm_model = cmd_args.model
    idx_persona = cmd_args.idx_persona
    which_format = cmd_args.format
    verbose = cmd_args.verbose
    save_only = cmd_args.save_only
    n_variants = cmd_args.n_variants

    base_dir = "./data/output"
    evaluation = Evaluation(args)
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(args['datasets']['random_contexts_file'], "r", encoding="utf-8") as f:
        all_irrelevant_contexts = json.load(f)

    if cmd_args.up_to is False:
        n_blocks = [cmd_args.n_blocks]
    else:
        n_blocks = range(1, cmd_args.n_blocks)

    for curr_n_blocks in n_blocks:
        if cmd_args.save_only:
            print(f"{utils.Colors.OKBLUE}Processing {curr_n_blocks} conversation blocks for persona_{idx_persona}{utils.Colors.ENDC}")
        else:
            output_file_path = f'./data/eval/{llm_model}_persona{idx_persona}_{curr_n_blocks}blocks.json'
            output_file_path_full_results = f'./data/eval/{llm_model}_persona{idx_persona}_{curr_n_blocks}blocks_full.json'
            print(f"{utils.Colors.OKBLUE}Evaluating {llm_model} on {curr_n_blocks} conversation blocks for persona_{idx_persona}{utils.Colors.ENDC}")
        results = {}
        full_results = []

        # Gather all candidate conversation blocks
        chosen_blocks, persona = load_n_conversation_blocks(idx_persona, curr_n_blocks, base_dir, verbose)

        # Process each chosen conversation block
        processed_blocks_dict = {}
        # all_strings = []
        # new_content_samples = [{} for _ in range(len(chosen_blocks))]

        for block_idx, ((file_name, time_period), conversation) in enumerate(chosen_blocks):
            topic = file_name.split('_')[1]
            # try:
            processed_conversation, latest_ts = process_conversation_block(topic, conversation, which_format)
            # except Exception as e:
            #     print(f"{utils.Colors.FAIL}Error processing conversation block {file_name}{utils.Colors.ENDC}")
            #     continue

            qa = extract_qa(base_dir, topic, file_name, time_period)

            if latest_ts in processed_blocks_dict:
                latest_ts = latest_ts + '_2'
            processed_blocks_dict[latest_ts] = {
                "conversation": processed_conversation[0],  # idx 0 corresponds to the conversation in the required format, either string or api_dict
                "file_name": file_name,
                "time_period": time_period,
                "last_timestamp": latest_ts,
                "topic": topic,
                "qa": qa
            }
            # all_strings.append(processed_conversation[-1])  # idx -1 always corresponds to the conversation in the plain string format

        # Topological sort chosen conversation blocks by the latest timestamp
        # print('Before sort new_content_samples: ', new_content_samples)
        variants = topological_sort(processed_blocks_dict, tokenizer, num_variants=n_variants, verbose=verbose)

        # Dictionary to store shared context IDs
        shared_context_id_set = set()

        for sorted_processed_blocks in variants:
            # Concatenate all conversation blocks
            all_conversations, num_irrelevant_tokens = concatenate_blocks(sorted_processed_blocks, which_format, tokenizer, all_irrelevant_contexts, persona, verbose)
            all_qa, all_conversations = add_all_qa_and_compute_distance(sorted_processed_blocks, tokenizer, all_conversations, num_irrelevant_tokens)

            total_num_tokens = count_tokens(" ".join([item['content'] for item in all_conversations if 'content' in item]), tokenizer, verbose=False)
            if verbose:
                print(f"{utils.Colors.OKGREEN}Number of tokens: {total_num_tokens} on gpt-4o tokenizer{utils.Colors.ENDC}")

            # Show all Q&As related to this concatenated conversation
            for (curr_context, question, formatted_question, correct_answer, all_options, distance_blocks, distance_tokens, question_type, topic, where, stereotypical,
                 context_length_in_tokens, context_length_in_letters, shared_context, end_index_in_shared_context, num_irrelevant_tokens) in tqdm(question_loader(all_qa), total=len(all_qa)):
                # Generate a random unique ID for the question
                question_id = str(uuid.uuid4())  # Generate a random unique ID
                shared_context_id = generate_shared_context_id(shared_context)    # More efficient way to store long context shared by multiple Q&As with just different end indices

                if save_only:
                    curr_qa_info = {
                        "question_id": question_id,
                        "idx_persona": idx_persona,
                        "question": question,
                        "correct_answer": correct_answer,
                        "all_options": all_options,
                        "distance_blocks": distance_blocks,
                        "distance_tokens": distance_tokens,
                        "question_type": question_type,
                        "topic": topic,
                        "shared_context_id": shared_context_id,
                        "end_index_in_shared_context": end_index_in_shared_context,
                        "context_length_in_tokens": context_length_in_tokens,
                        "context_length_in_letters": context_length_in_letters,
                        "num_irrelevant_tokens": num_irrelevant_tokens,
                        "stereotypical": stereotypical,
                    }
                    # Save the contexts to JSON and the question-answer pairs to CSV as our released dataset
                    save_contexts_to_json({question_id: curr_context}, f"data/contexts_{benchmark_size}.jsonl")
                    if shared_context_id not in shared_context_id_set:
                        save_contexts_to_json({shared_context_id: shared_context}, f"data/shared_contexts_{benchmark_size}.jsonl")
                        shared_context_id_set.add(shared_context_id)
                    save_questions_to_csv(curr_qa_info, f"data/questions_{benchmark_size}.csv")


                else:
                    predicted_answer = evaluation.query_llm(curr_context, formatted_question, which_format)
                    match = evaluate_answer(predicted_answer, correct_answer)

                    if verbose:
                        print(f'{utils.Colors.OKGREEN}{"Correct answer"}:{utils.Colors.ENDC}{correct_answer}')
                        print(f'{utils.Colors.OKGREEN}{"Predicted answer"}:{utils.Colors.ENDC}{predicted_answer}')
                        if match:
                            print(f'{utils.Colors.OKBLUE}{"Correct"}{utils.Colors.ENDC}')
                        else:
                            print(f'{utils.Colors.FAIL}{"Incorrect"}{utils.Colors.ENDC}')

                    """
                    (1) Save evaluation results based on the distances from the question being asked at the end to the sourced conversation block
                    (2) Save results based on the question types
                    (3) Save results based on the conversation contexts
                    (4) Evaluation with long contexts is expensive, so we also save full results for further analysis
                    """
                    keys = [distance_blocks, distance_tokens, question_type, context]
                    for key in keys:
                        if key == distance_blocks:
                            key = f"distance_blocks_{key}"
                        if key == distance_tokens:
                            key = f"distance_tokens_{key}"
                        if key not in results:
                            results[key] = {"correct": 0, "total": 0}
                        else:
                            results[key]["total"] += 1
                            if match:
                                results[key]["correct"] += 1

                    full_results.append({
                            "question_id": question_id,
                            "question": formatted_question,
                            "correct_answer": correct_answer,
                            "all_options": all_options,
                            "predicted_answer": predicted_answer,
                            "match": match,
                            "distance_blocks": distance_blocks,
                            "distance_tokens": distance_tokens,
                            "question_type": question_type,
                            "topic": topic
                        }
                    )

            shared_contexts = read_jsonl_file(f"data/shared_contexts_{benchmark_size}.jsonl")
            contexts = read_jsonl_file(f"data/contexts_{benchmark_size}.jsonl")
            print(f"Number of contexts: {len(contexts)}")
            print(f"Number of shared contexts: {len(shared_contexts)}")

            if not save_only:
                # Calculate the percentage of the results
                for key in results:
                    results[key]["accuracy"] = results[key]["correct"] / results[key]["total"] * 100 if results[key]["total"] > 0 else 0
                print(f'{utils.Colors.OKGREEN}{"Final Results"}:{utils.Colors.ENDC}')
                for key in results:
                    print(f'{key}: {results[key]["accuracy"]:.2f}%')

                # Save evaluation results to a JSON file.
                with open(output_file_path, "w") as json_file:
                    json.dump(results, json_file, indent=4)

                full_results.append({"all_conversations": all_conversations})
                with open(output_file_path_full_results, "w") as json_file:
                    json.dump(full_results, json_file, indent=4)

