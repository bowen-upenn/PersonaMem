import json
import os
import sys

import numpy as np
import pandas as pd
from chat_lib import chatSession, split_on_system
from data_loader import chunks, data_loader
from FlagEmbedding import BGEM3FlagModel
from mem0_lib import build_config
from tqdm import tqdm

from mem0 import Memory

MEM0_CHUNK_SIZE = 64
USER_ID = "123"
cache = {}


def reset_memory(memory):
    if memory is None:
        return
    if len(memory.get_all(USER_ID)["results"]) > 0:
        memory.reset()


def encode_with_cache(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    new_texts = [t for t in texts if t not in cache]
    if new_texts:
        new_embeddings = model.encode(new_texts)["dense_vecs"]
        for text, emb in zip(new_texts, new_embeddings):
            cache[text] = emb
    return np.array([cache[t] for t in texts])


def evaluator(
    questions,
    contexts,
    emb_model,
    openai_model,
    openai_api_key,
    top_k=5,
    context_mode="none",
    idx_shard=-1,
):
    predictions = []
    choosen_indices = []

    memory = None

    for row in tqdm(questions.iterrows(), total=questions.shape[0]):
        print("processing row", row[0])

        target_question = row[1].get("question") or row[1].get("user_question_or_message")
        target_options = row[1]["all_options"]
        target_options = "\n".join(target_options)
        if context_mode != "none":
            context_id = row[1]["shared_context_id"]
            end_index = row[1]["end_index_in_shared_context"]
            target_context = contexts[context_id][0:end_index]
            target_context = [str(x) for x in target_context]
            # previously x['context'] only, but now we include 'role' and 'context' together

        session = chatSession(model=openai_model, openai_api_key=openai_api_key)

        if context_mode == "rag":
            embeddings_1 = encode_with_cache(emb_model, target_question)
            embeddings_2 = encode_with_cache(emb_model, target_context)
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
            suffix = idx_shard if idx_shard != -1 else ""
            mem0_config = build_config(
                openai_model,
                openai_api_key,
                collection_name=f"Question{row[0]}",
                vector_store="chroma",
                suffix=suffix,
            )

            memory = Memory.from_config(mem0_config)

            # TODO: data loader actually does json.dumps, so could refactor to avoid extra work
            msg_history_all = [json.loads(turn) for turn in target_context]
            msg_histories = split_on_system(msg_history_all)
            for msg_history in tqdm(msg_histories, desc="Adding each session to memory"):
                memory.add(msg_history, user_id=USER_ID)  # only retrieves from user history

            mems_relevant = memory.search(target_question, user_id=USER_ID, limit=top_k)["results"]
            mems_flat = "\n".join([m["memory"] for m in mems_relevant])
            prompt = f"# User facts: \n{mems_flat}\n\n"
            top_k_idx = mems_flat

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

    top_k = sys.argv[1]
    context_mode = sys.argv[2]
    gpt_model = sys.argv[3]

    idx_shard = -1
    num_shards = 1
    if len(sys.argv) > 5:
        idx_shard = int(sys.argv[4])
        num_shards = int(sys.argv[5])

    PATH_questions = "data/questions.csv"
    if not os.path.exists(PATH_questions):
        print(f"Error: {PATH_questions} must exist")
        sys.exit(1)

    PATH_contexts = "data/contexts.json" if not context_mode == "none" else None
    if PATH_contexts and not os.path.exists(PATH_contexts):
        print(f"Error: {PATH_contexts} must exist")
        sys.exit(1)

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

    emb_model = None
    if context_mode == "rag":
        emb_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("Embedding Model loaded successfully")
    elif context_mode == "mem0":
        print(
            "You may see some errors message from mem0, since it's using prompt engineering and JSONs are not always valid. You can probably ignore them."
        )

    questions, contexts = data_loader(PATH_questions, PATH_contexts, fix_json=False)
    print("Data loaded successfully")
    if num_shards > 1:
        dataset_rows_sharded = list(chunks(questions, num_shards))
        num_qs_total = len(questions)
        questions = dataset_rows_sharded[idx_shard]
        print(
            f"Processing shard {idx_shard + 1}/{num_shards} with {len(questions)}/{num_qs_total} questions"
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    predictions, choosen_indices = evaluator(
        questions, contexts, emb_model, gpt_model, openai_api_key, top_k, context_mode, idx_shard
    )

    # calculate accuracy
    correct_answer = questions["correct_answer"].apply(lambda x: x[1])
    predictions = [x.lower() for x in predictions]
    list_correct = [correct_answer.iloc[i] == predictions[i] for i in range(len(predictions))]
    accuracy = sum(list_correct) / len(list_correct)
    print(f"Accuracy: {accuracy}")

    # output results
    df_prediction = pd.DataFrame()
    df_prediction["question_id"] = questions["question_id"]
    df_prediction["answer"] = correct_answer
    df_prediction["p"] = predictions
    df_prediction["r"] = choosen_indices

    stem = f"pred_k={top_k}_r={context_mode}_m={gpt_model}"
    if num_shards > 1:
        stem += f"_{idx_shard + 1}-{num_shards}"
    df_prediction.to_csv(f"data/{stem}.csv", index=False)
    print("Predictions saved successfully")
