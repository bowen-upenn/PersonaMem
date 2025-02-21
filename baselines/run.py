import os
import sys

import numpy as np
import pandas as pd
from chat_lib import chatSession
from data_loader import data_loader
from FlagEmbedding import BGEM3FlagModel
from mem0_lib import build_config
from tqdm import tqdm

from mem0 import Memory


def reset_memory(memory):
    if memory is None:
        return
    memory.reset()


def evaluator(
    questions, contexts, emb_model, openai_model, openai_api_key, top_k=5, context_mode="none"
):
    predictions = []
    choosen_indices = []

    memory = None
    if context_mode == "mem0":
        mem0_config = build_config(openai_model, openai_api_key)

        memory = Memory.from_config(mem0_config)

    for row in tqdm(questions.iterrows(), total=questions.shape[0]):
        reset_memory(memory)  # reset memory for each question

        target_context = contexts[row[1]["question_id"]]
        target_context = [x["content"] for x in target_context]
        target_question = row[1]["question"]
        target_options = row[1]["all_options"]
        target_options = "\n".join(target_options)

        session = chatSession(model=openai_model, openai_api_key=openai_api_key)

        if context_mode == "rag":
            embeddings_1 = emb_model.encode([target_question])["dense_vecs"]
            embeddings_2 = emb_model.encode(target_context)["dense_vecs"]
            similarity = embeddings_1 @ embeddings_2.T

            # index of top k similar context

            top_k_idx = np.argsort(similarity[0])[::-1][:top_k]
            retrieved_context = [target_context[i] for i in top_k_idx]
            retrieved_context = "\n".join(retrieved_context)

            prompt = f"# {top_k} most similar contexts to the question: \n{retrieved_context}\n\n"
        elif context_mode == "none":
            top_k_idx = None
            prompt = ""
        elif context_mode == "all":
            top_k_idx = None
            prompt = ""

            role = "user"
            for context in target_context:
                session.add_message(role, context)
                role = "assistant" if role == "user" else "user"
        elif context_mode == "mem0":
            role = "user"
            for context in target_context:
                session.add_message(role, context)
                role = "assistant" if role == "user" else "user"
            msg_history = session.msg_history
            memory.add(msg_history, user_id="123")  # only retrieves from user history
            mems_relevant = memory.search(target_question, user_id="123", limit=top_k)["results"]
            mems_flat = "\n".join([m["memory"] for m in mems_relevant])
            prompt = f"# User facts: \n{mems_flat}\n\n"
            top_k_idx = mems_flat
            session.clear_history()

        prompt += f"# Question: {target_question}\n\n"
        prompt += f"# Choices: \n{target_options}"

        session.add_message("user", prompt)
        response = session.get_message()

        predictions.append(response["answer"])
        choosen_indices.append(top_k_idx)

    return predictions, choosen_indices


if __name__ == "__main__":
    # Ensure the required arguments are provided
    if len(sys.argv) < 3:
        print(
            "Error: Missing required arguments. Please provide values for 'top_k', 'context_mode' and 'gpt_model."
        )
        print("Usage: python script.py <top_k> <context_mode> <gpt_model>")
        sys.exit(1)

    if not os.path.exists("data/questions.csv") or not os.path.exists("data/contexts.json"):
        print("Error: questions.csv and contexts.json must exist in ./data")
        sys.exit(1)

    top_k = sys.argv[1]
    context_mode = sys.argv[2]
    gpt_model = sys.argv[3]

    # Validate top_k: Ensure it is a positive integer
    if not top_k.isdigit() or int(top_k) <= 0:
        print("Error: 'top_k' must be a positive integer.")
        sys.exit(1)

    top_k = int(top_k)  # Convert to integer after validation

    # Validate context_mode
    MODES = {"none", "rag", "mem0", "all"}
    if context_mode.lower() not in MODES:
        print(f"Error: 'context_mode' must be one of {MODES}.")
        sys.exit(1)

    # validata gpt_model: Ensure it is a valid model
    if gpt_model not in {"gpt-4o", "gpt-4o-mini"}:
        print("Error: 'gpt_model' must be either 'gpt-4o' or 'gpt-4o-mini'.")
        sys.exit(1)

    print(f"top_k is set to {top_k}")
    print(f"context_mode is set to {context_mode}")
    print(f"gpt_model is set to {gpt_model}")

    PATH_questions = "data/questions.csv"
    PATH_contexts = "data/contexts.json"

    emb_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("Embedding Model loaded successfully")
    questions, contexts = data_loader(PATH_questions, PATH_contexts, fix_json=False)
    print("Data loaded successfully")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    predictions, choosen_indices = evaluator(
        questions, contexts, emb_model, gpt_model, openai_api_key, top_k, context_mode
    )

    # calculate accuracy
    correct_answer = questions["correct_answer"].apply(lambda x: x[1])
    predictions = [x.lower() for x in predictions]
    list_correct = [correct_answer[i] == predictions[i] for i in range(len(predictions))]
    accuracy = sum(list_correct) / len(list_correct)
    print(f"Accuracy: {accuracy}")

    # output results
    df_prediction = pd.DataFrame()
    df_prediction["question_id"] = questions["question_id"]
    df_prediction["answer"] = correct_answer
    df_prediction["p"] = predictions
    df_prediction["r"] = choosen_indices

    df_prediction.to_csv(f"data/pred_k={top_k}_r={context_mode}_m={gpt_model}.csv", index=False)
    print("Predictions saved successfully")
