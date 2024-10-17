import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re
import math

from openai import OpenAI

import prompts
import utils


class QueryLLM:
    def __init__(self, args):
        self.args = args
        # Load the API key
        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()

        self.client = OpenAI(api_key=self.api_key)
        self.assistant = self.client.beta.assistants.create(
            name="Data Generator",
            instructions="You are a helpful assistant that generates persona-oriented conversational data in an user specified context.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm_model'],
        )
        self.thread = None
        self.expanded_persona = None
        self.init_general_personal_history = None
        self.expanded_general_personal_history = None

    def create_a_thread(self):
        self.thread = self.client.beta.threads.create()

    def query_llm(self, step='source_data', content=None, context=None, idx_context=0, verbose=False):
        if step == 'source_data':
            prompt = prompts.prompts_for_background_data(content)
        elif step == 'expand_persona':
            prompt = prompts.prompts_for_expanding_persona(content)
        elif step == 'init_general_personal_history':
            prompt = prompts.prompts_for_init_general_personal_history(content)
        elif step == 'init_contextual_personal_history':
            prompt = prompts.prompts_for_init_contextual_personal_history(context, self.expanded_persona, self.init_general_personal_history)
        elif step == 'init_conversation':
            if context == 'therapy':
                prompt = prompts.prompts_for_init_therapy_conversations()
            else:
                raise NotImplementedError("Unknown context: {}".format(context))
        elif step == 'generate_questions':
            prompt = prompts.prompt_for_question_answer_pairs()
        elif step == 'expand_history_and_conversation':
            if context == 'therapy':
                prompt = prompts.prompts_for_second_general_personal_history_and_therapy_conversations(context, self.expanded_general_personal_history)
            else:
                raise NotImplementedError("Unknown context: {}".format(context))
        else:
            raise ValueError(f'Invalid step: {step}')

        if step == 'expand_persona':
            response = self.client.chat.completions.create(
                model=self.args['models']['llm_model'],
                messages=[{"role": "user",
                           "content": prompt}],
                max_tokens=300
            )
            response = response.choices[0].message.content
            if verbose:
                print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
        else:
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
                response = response.data[0].content[0].text.value
                if verbose:
                    print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
            else:
                response = None
                print(run.status)

        if idx_context == 0 and step == 'expand_persona':
            self.expanded_persona = response
        if idx_context == 0 and step == 'init_general_personal_history':
            self.init_general_personal_history = response
        if idx_context == 0 and step == 'second_expand':
            self.expanded_general_personal_history = response

        return response
