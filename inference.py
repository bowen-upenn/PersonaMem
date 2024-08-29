from tqdm import tqdm
import os
import numpy as np
import re
import json
import random
import torch

# from query_llm import QueryLLM
from utils import *


def inference(args):
    # LLM = QueryLLM(args)

    with torch.no_grad():
        for idx_sample in range(int(args['inference']['num_samples'])):
            # Load all personas
            with open(args['datasets']['persona_file'], 'r') as file:
                all_personas = file.readlines()

            ############### Step 1: Load a random conversation history from the chosen real-world dataset ###############
            if args['datasets']['context'] == 'therapy':
                source_dir = args['datasets']['therapy_source_dir']
            else:
                raise NotImplementedError("Unknown context: {}".format(args['datasets']['context']))

            all_files = os.listdir(source_dir)
            random_idx = random.randint(0, len(all_files) - 1)
            selected_file = all_files[random_idx]
            selected_file_path = os.path.join(source_dir, selected_file)

            with open(selected_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Convert JSON to plain text
            context_conversation = ""
            for message in data["conversation"]:
                role = message["role"]
                content = message["content"]
                context_conversation += f"{role.capitalize()}: {content}\n\n"

            # # Send the conversation to the LLM as a background memory about the context
            _ = LLM.query_llm(query=context_conversation, step='source_data', verbose=args['inference']['verbose'])

            ############### Step 2: Load a random persona ###############
            random_row = random.choice(all_personas)
            print(random_row.strip())
            persona = random_row.strip()[13:-2]  # Remove {"persona": "} and "}

            ############### Step 3: Expand the persona to personal history ###############


            ############### Step 4: Expand the persona and personal history to conversation ###############


            ############### Step 5: Generate a list of memory-related questions and answers ###############


            ############### Step 6: Continue writing new personal history with conflicts and another conversation ###############


            ############### Step 7: Continue generating another list of memory-related questions and answers ###############
