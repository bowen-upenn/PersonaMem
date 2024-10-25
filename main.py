import numpy as np
import torch
import yaml
import os
import json
import argparse
import sys

from inference import inference


if __name__ == "__main__":
    print("Python", sys.version, 'Torch', torch.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--context', type=str, default="therapy", nargs="+", help='Set conversation context. Choose from therapy. If multiple contexts, sperate the names by space, e.g. --context therapy legal')  # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument('--n_persona', type=int, default=1, help='Set number of personas to generate')
    parser.add_argument('--n_samples', type=int, default=1, help='Set number of samples per context to generate')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['datasets']['context'] = cmd_args.context if cmd_args.context is not None else args['datasets']['context']
    args['inference']['num_personas'] = cmd_args.n_persona if cmd_args.n_persona is not None else args['inference']['num_personas']
    args['inference']['num_samples_per_context'] = cmd_args.n_samples if cmd_args.n_samples is not None else args['inference']['num_samples_per_context']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Start inference
    print(args)
    inference(args)
