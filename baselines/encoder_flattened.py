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
    
    list_text = []
    pointer_list = [0]
    pointer = 0
    for key in contexts.keys():
        incoming_list = contexts[key]
        list_text.extend(incoming_list)
        pointer = pointer + len(incoming_list)
        pointer_list.append(pointer)

    emb_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("Embedding Model loaded successfully")

    list_context_emb = emb_model.encode(list_text)['dense_vecs']
    print("Contexts encoded successfully")

    with open(f"{PATH}context_embs_list.pkl", "wb") as f:
        pickle.dump(list_context_emb, f)


    dict_context_emb = {}
    for key_id, key in enumerate(list(contexts.keys())):
        start = pointer_list[key_id]
        end = pointer_list[key_id+1]
        dict_context_emb[key] = list_context_emb[start:end]


    # Save embeddings as .npz
    np.savez_compressed(f"{PATH}context_embs.npz", **dict_context_emb)
    print(f"Context embeddings saved to {PATH}context_embs.npz")

    # To use:
    # loaded_data = np.load(f"{PATH}/context_embs.npz", allow_pickle=True)
    # context_embs = {key: loaded_data[key] for key in loaded_data.files}


if __name__ == "__main__":
    # listen for the path
    import sys
    if len(sys.argv) < 2:
        print("Usage: python encoder.py <path>")
        sys.exit(1)

    PATH = sys.argv[1]
    
    # Run the encoding function
    encode_contexts(PATH)