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
        self.assistant_persona = self.client.beta.assistants.create(
            name="Persona Generator",
            instructions="You are a helpful assistant that generates user personas in an user specified context.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm_model'],
        )
        self.thread_persona = None

        self.assistant_conversation = self.client.beta.assistants.create(
            name="Conversation Generator",
            instructions="You are a helpful assistant that generates persona-oriented conversational data in an user specified context.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm_model'],
        )
        self.thread_conversation = None
        self.thread_preparing_new_content = None
        self.thread_new_content = None
        self.thread_eval_new_content = None

        self.expanded_persona = None

        self.general_personal_history = None
        self.init_general_personal_history = None
        self.first_expand_general_personal_history = None
        self.second_expand_general_personal_history = None
        self.third_expand_general_personal_history = None

        self.init_personal_history = None
        self.first_expand_personal_history = None
        self.second_expand_personal_history = None
        self.third_personal_history = None

    def create_a_thread(self, step):
        if step == 'conversation':
            self.thread_persona = self.client.beta.threads.create()
            self.thread_conversation = self.client.beta.threads.create()
            self.thread_preparing_new_content = self.client.beta.threads.create()
        elif step == 'qa':
            self.thread_new_content = self.client.beta.threads.create()
            self.thread_eval_new_content = self.client.beta.threads.create()
        else:
            raise ValueError(f'Invalid step: {step}')

    def query_llm(self, step='source_data', persona=None, context=None, seed=None, data=None, action=None, idx_context=0, start_time=None, verbose=False):
        if step == 'source_data':
            prompt = prompts.prompts_for_background_data(seed)
        elif step == 'expand_persona':
            prompt = prompts.prompts_for_expanding_persona(persona, start_time)

        # Generate once across multiple contexts
        elif step == 'init_general_personal_history':
            prompt = prompts.prompts_for_init_general_personal_history(persona, start_time)
        elif step == 'first_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='WEEK')
        elif step == 'second_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='MONTH')
        elif step == 'third_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='YEAR')

        # Generate one for each context
        elif step == 'init_contextual_personal_history':
            prompt = prompts.prompts_for_init_contextual_personal_history(context, start_time, self.expanded_persona, self.general_personal_history)
        elif step == 'first_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(context=context, type='general', period='WEEK')
        elif step == 'second_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(context=context, type='general', period='MONTH')
        elif step == 'third_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(context=context, type='general', period='YEAR')

        # A separate thread to populate personal histories into conversations
        elif step == 'init_conversation':
            prompt = prompts.prompts_for_generating_conversations(context, self.expanded_persona, curr_personal_history=self.init_personal_history, period='INIT')
        elif step == 'first_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(context, self.expanded_persona, curr_personal_history=self.first_expand_personal_history, period='WEEK')
        elif step == 'second_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(context, self.expanded_persona, curr_personal_history=self.second_expand_personal_history, period='MONTH')
        elif step == 'third_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(context, self.expanded_persona, curr_personal_history=self.third_personal_history, period='YEAR')

        elif step == 'qa_helper':
            prompt = prompts.prompts_for_generating_qa(data, action)
        elif step == 'prepare_new_content':
            prompt = prompts.prompt_for_preparing_new_content(data, action)
        elif step == 'new_content':
            prompt = prompts.prompt_for_content_generation(data, action)
        elif step == 'eval_new_content':
            prompt = prompts.prompt_for_evaluating_content(data, action)
        else:
            raise ValueError(f'Invalid step: {step}')

        # Independent API calls every time
        if step == 'expand_persona' or step == 'qa_helper':
            response = self.client.chat.completions.create(
                model=self.args['models']['llm_model'],
                messages=[{"role": "user",
                           "content": prompt}]
            )
            response = response.choices[0].message.content
            if verbose:
                print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')

        # API calls within a thread in a multi-turn fashion
        else:
            if step == 'source_data' or step == 'init_conversation' or step == 'first_expand_conversation' or step == 'second_expand_conversation' or step == 'third_expand_conversation':
                curr_thread = self.thread_conversation
            elif step == 'prepare_new_content':
                curr_thread = self.thread_preparing_new_content
            elif step == 'new_content':
                curr_thread = self.thread_new_content
            elif step == 'eval_new_content':
                curr_thread = self.thread_eval_new_content
            else:
                curr_thread = self.thread_persona

            message = self.client.beta.threads.messages.create(
                thread_id=curr_thread.id,
                role="user",
                content=prompt
            )

            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=curr_thread.id,
                assistant_id=self.assistant_persona.id
            )

            if run.status == 'completed':
                response = self.client.beta.threads.messages.list(
                    thread_id=curr_thread.id
                )
                response = response.data[0].content[0].text.value
                if verbose:
                    if step == 'new_content':
                        print(f'{utils.Colors.OKGREEN}{action.capitalize()}:{utils.Colors.ENDC} {response}')
                    else:
                        print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
            else:
                response = None
                print(run.status)

        # Save general personal history to be shared across contexts
        if idx_context == 0:
            if step == 'init_general_personal_history':
                self.general_personal_history = response
                self.init_general_personal_history = response
            elif step == 'first_expand_general_personal_history':
                self.general_personal_history += response
                self.first_expand_general_personal_history = response
            elif step == 'second_expand_general_personal_history':
                self.general_personal_history += response
                self.second_expand_general_personal_history = response
            elif step == 'third_expand_general_personal_history':
                self.general_personal_history += response
                self.third_expand_general_personal_history = response
            elif step == 'expand_persona':
                self.expanded_persona = response

        # Save general+contextual personal history in order to generate conversations
        if step == 'init_general_personal_history':
            self.init_personal_history = response
        elif step == 'init_contextual_personal_history':
            self.init_personal_history += response
        elif step == 'first_expand_general_personal_history':
            self.first_expand_personal_history = response
        elif step == 'first_expand_contextual_personal_history':
            self.first_expand_personal_history += response
        elif step == 'second_expand_general_personal_history':
            self.second_expand_personal_history = response
        elif step == 'second_expand_contextual_personal_history':
            self.second_expand_personal_history += response
        elif step == 'third_expand_general_personal_history':
            self.third_personal_history = response
        elif step == 'third_expand_contextual_personal_history':
            self.third_personal_history += response

        return response
