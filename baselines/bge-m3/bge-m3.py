from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import os
import json
import numpy as np
from openai import OpenAI
import os
from tqdm import tqdm
import sys

from data_loader import data_loader

class chatSession:
    def __init__(self, messages = None, openai_api_key = None, model = 'gpt-4o-mini', temperature = 0):
        self.system_msg = [{"role": "system", "content": "You are an AI that answers only with A, B, C, or D."}]
        self.msg_history = []
        self.openai_client = OpenAI(api_key=openai_api_key)

        if messages is not None:
            for idx, msg in enumerate(messages):
                self.msg_history.append(msg)
        self.model = model
        self.temperature = temperature
        self.json_schema = {
                "name": "answer_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"]
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                }
            }

    def add_message(self, role, content):
        self.msg_history.append({"role": role, "content": content})

    def update_system_msg(self, system_msg):
        self.system_msg = [{"role": "system", "content": system_msg}]



    def get_message(self, response_format = 'json_schema', update_history = True, **kwargs):
        input_messages = self.system_msg + self.msg_history
        input_messages = [{"role": message["role"], "content": str(message["content"])} for message in input_messages]
        response = self.openai_client.chat.completions.create(
            model=self.model,
            response_format = {"type": response_format, "json_schema": self.json_schema},
            messages= input_messages,
            temperature=self.temperature,
        )
        output_msg = response.choices[0].message.content
        if response_format == 'json_schema':
            try:
                output_msg = json.loads(output_msg)
            except Exception as e:
                print(f"Invalid JSON format from OpenAI. Error: {e}.")
                print(output_msg)
                return self.get_message(response_format = 'json_schema', update_history = True, **kwargs)
        if update_history:
            self.msg_history.append({"role": "assistant", "content": output_msg})
        return output_msg

def evaluator(questions, contexts, emb_model, openai_model, openai_api_key, top_k = 5, do_retrieval = True):
    predictions = []
    choosen_indices = []

    
    for row in tqdm(questions.iterrows(), total = questions.shape[0]):
        target_context = contexts[row[1]['question_id']]
        target_context = [x['content'] for x in target_context]
        target_question = row[1]['question']
        target_options = row[1]['all_options']
        target_options = "\n".join(target_options)

        session = chatSession(model = openai_model, openai_api_key = openai_api_key)

        if do_retrieval:
            embeddings_1 = emb_model.encode([target_question])['dense_vecs']
            embeddings_2 = emb_model.encode(target_context)['dense_vecs']
            similarity = embeddings_1 @ embeddings_2.T

            # index of top k similar context
            
            top_k_idx = np.argsort(similarity[0])[::-1][:top_k]
            retrieved_context = [target_context[i] for i in top_k_idx]
            retrieved_context = '\n'.join(retrieved_context)

            

            prompt = f"# {top_k} most similar contexts to the question: \n{retrieved_context}\n\n"
            prompt += f"# Question: {target_question}\n\n"
            prompt += f"# Choices: \n{target_options}"

        else:
            top_k_idx = None
            prompt = f"# Question: {target_question}\n\n"
            prompt += f"# Choices: \n{target_options}"

        session.add_message("user", prompt)
        response = session.get_message()

        predictions.append(response['answer'])
        choosen_indices.append(top_k_idx)

    return predictions, choosen_indices


if __name__ == '__main__':
    # Ensure the required arguments are provided
    if len(sys.argv) < 3:
        print("Error: Missing required arguments. Please provide values for 'top_k', 'do_retrieval' and 'gpt_model.")
        print("Usage: python script.py <top_k> <do_retrieval> <gpt_model>")
        sys.exit(1)

    if not os.path.exists("data/questions.csv") or not os.path.exists("data/contexts.json"):
        print("Error: questions.csv and contexts.json must exist in ./data")
        sys.exit(1)

    top_k = sys.argv[1]
    do_retrieval = sys.argv[2]
    gpt_model = sys.argv[3]

    # Validate top_k: Ensure it is a positive integer
    if not top_k.isdigit() or int(top_k) <= 0:
        print("Error: 'top_k' must be a positive integer.")
        sys.exit(1)

    top_k = int(top_k)  # Convert to integer after validation

    # Validate do_retrieval: Ensure it is 'true' or 'false' (case insensitive)
    if do_retrieval.lower() not in {"true", "false"}:
        print("Error: 'do_retrieval' must be either 'true' or 'false'.")
        sys.exit(1)

    # validata gpt_model: Ensure it is a valid model
    if gpt_model not in {"gpt-4o", "gpt-4o-mini"}:
        print("Error: 'gpt_model' must be either 'gpt-4o' or 'gpt-4o-mini'.")
        sys.exit(1)

    do_retrieval = do_retrieval.lower() == "true"  # Convert to boolean

    print(f"top_k is set to {top_k}")
    print(f"do_retrieval is set to {do_retrieval}")
    print(f"gpt_model is set to {gpt_model}")

    PATH_questions = "data/questions.csv"
    PATH_contexts = "data/contexts.json"

    emb_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 
    print("Embedding Model loaded successfully")
    questions, contexts = data_loader(PATH_questions, PATH_contexts, fix_json = False)
    print("Data loaded successfully")


    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    predictions, choosen_indices = evaluator(questions, contexts, emb_model, gpt_model, openai_api_key, top_k, do_retrieval)

    # calculate accuracy
    correct_answer = questions['correct_answer'].apply(lambda x: x[1])
    predictions = [x.lower() for x in predictions]
    list_correct = [correct_answer[i] == predictions[i] for i in range(len(predictions))]
    accuracy = sum(list_correct)/len(list_correct)
    print(f"Accuracy: {accuracy}")

    # output results
    df_prediction = pd.DataFrame()
    df_prediction['question_id'] = questions['question_id']
    df_prediction['answer'] = correct_answer
    df_prediction['p'] = predictions
    df_prediction['r'] = choosen_indices

    df_prediction.to_csv(f'data/pred_k={top_k}_r={do_retrieval}_m={gpt_model}.csv', index = False)
    print("Predictions saved successfully")
