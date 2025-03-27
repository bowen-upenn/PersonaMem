import os
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from data_loader import data_loader
import pickle

def encode_contexts(PATH):
    if not PATH.endswith('/'):
        PATH += '/'
    if not os.path.exists(PATH):
        raise FileNotFoundError(f"The path {PATH} does not exist.")

    PATH_questions = PATH + "questions.csv"
    PATH_contexts = PATH + "shared_contexts.jsonl"
    
    if not os.path.exists(PATH_questions) or not os.path.exists(PATH_contexts):
        raise FileNotFoundError(f"Required files not found in the path {PATH}. Please ensure 'questions.csv' and 'shared_contexts.jsonl' are present.")
    
    questions, contexts = data_loader(PATH_questions, PATH_contexts, fix_json=False)
    print("Data loaded successfully")
    
    list_text = questions['user_question_or_message'].tolist()

    emb_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("Embedding Model loaded successfully")

    list_context_emb = emb_model.encode(list_text)['dense_vecs']
    print("Questions encoded successfully")

    with open(f"{PATH}question_embs_list.pkl", "wb") as f:
        pickle.dump(list_context_emb, f)

    print(f"Question embeddings saved to {PATH}question_embs_list.pkl")


if __name__ == "__main__":
    # listen for the path
    import sys
    if len(sys.argv) < 2:
        print("Usage: python encoder.py <path>")
        sys.exit(1)

    PATH = sys.argv[1]
    
    # Run the encoding function
    encode_contexts(PATH)