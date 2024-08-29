import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re
import math

from openai import OpenAI

from prompts import *
from utils import *


class QueryLLM:
    def __init__(self, args):
        self.args = args
        # Load the API key
        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()

        self.client = OpenAI()
        self.assistant = client.beta.assistants.create(
            name="Data Generator",
            instructions="You are a helpful assistant that generates persona-oriented conversational data in an user specified context.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm_model'],
        )
        self.thread = client.beta.threads.create()


    def query_llm(self, query, step='source_data', verbose=False):
        if step == 'source_data':
            prompt = ""
        else:
            raise ValueError(f'Invalid step: {step}')

        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=prompt
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )

        if run.status == 'completed':
            response = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            if verbose:
                print(f'LLM Response: {response}')
        else:
            response = None
            print(run.status)

        return response
