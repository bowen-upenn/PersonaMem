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
            name="Conversation Generator",
            instructions="You are a helpful assistant that generates persona-oriented conversational data in an user specified topic.",
            tools=[{"type": "code_interpreter"}],
            model=self.args['models']['llm_model'],
        )
        self.assistant_irrelevant = self.client.beta.assistants.create(
            name="Conversation Generator",
            instructions="You are a helpful assistant.",
            model="gpt-4o-mini",
        )

        self.thread_irrelevant = None
        self.thread_persona = None
        self.thread_conversation = None
        self.thread_reflect_conversation = None
        self.thread_preparing_new_content = None
        self.thread_new_content = None
        self.thread_eval_new_content = None

        self.expanded_persona = ""

        self.general_personal_history = ""
        self.init_general_personal_history = ""
        self.first_expand_general_personal_history = ""
        self.second_expand_general_personal_history = ""
        self.third_expand_general_personal_history = ""

        self.init_personal_history = ""
        self.first_expand_personal_history = ""
        self.second_expand_personal_history = ""
        self.third_expand_personal_history = ""

    def create_a_thread(self, step):
        if step == 'conversation':
            self.thread_persona = self.client.beta.threads.create()
            self.thread_conversation = self.client.beta.threads.create()
            self.thread_reflect_conversation = self.client.beta.threads.create()
            self.thread_preparing_new_content = self.client.beta.threads.create()
        elif step == 'qa':
            self.thread_new_content = self.client.beta.threads.create()
            self.thread_eval_new_content = self.client.beta.threads.create()
        elif step == 'irrelevant':
            self.thread_irrelevant = self.client.beta.threads.create()
        else:
            raise ValueError(f'Invalid step: {step}')

    def delete_a_thread(self, step):
        if step == 'conversation':
            self.client.beta.threads.delete(thread_id=self.thread_persona.id)
            self.client.beta.threads.delete(thread_id=self.thread_conversation.id)
            self.client.beta.threads.delete(thread_id=self.thread_reflect_conversation.id)
            self.client.beta.threads.delete(thread_id=self.thread_preparing_new_content.id)
        elif step == 'qa':
            self.client.beta.threads.delete(thread_id=self.thread_new_content.id)
            self.client.beta.threads.delete(thread_id=self.thread_eval_new_content.id)
        elif step == 'irrelevant':
            self.client.beta.threads.delete(thread_id=self.thread_irrelevant.id)
        else:
            raise ValueError(f'Invalid step: {step}')

    def query_llm(self, step='source_data', persona=None, topic=None, seed=None, data=None, action=None, data_type=None, idx_topic=0, start_time=None, verbose=False):
        if step == 'source_data':
            prompt = prompts.prompts_for_background_data(seed)
        elif step == 'elaborate_topic':
            prompt = prompts.prompts_for_elaborating_topic(topic)
        elif step == 'expand_persona':
            prompt = prompts.prompts_for_expanding_persona(persona, start_time)

        elif step == 'random_question':
            prompt = data + " Explain thoroughly in details. "
        elif step == 'random_question_follow_up':
            prompt = prompts.prompts_for_random_question_follow_up()
        elif step == 'random_question_follow_up_response':
            prompt = data + " Explain thoroughly in details. "

        elif step == 'translate_code':
            prompt = prompts.prompts_for_translating_code(data, persona)
        elif step == 'rewrite_email':
            prompt = prompts.prompts_for_rewriting_email(data, persona)
        elif step == 'rewrite_creative_writing':
            prompt = prompts.prompts_for_rewriting_creative_writing(data, persona)

        # Generate once across multiple contexts
        elif step == 'init_general_personal_history':
            prompt = prompts.prompts_for_init_general_personal_history(persona, start_time)
        elif step == 'first_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='WEEK')
        elif step == 'second_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='MONTH')
        elif step == 'third_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='YEAR')

        # Generate one for each topic
        elif step == 'init_contextual_personal_history':
            prompt = prompts.prompts_for_init_contextual_personal_history(topic, start_time, self.expanded_persona, self.general_personal_history)
        elif step == 'first_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='WEEK')
        elif step == 'second_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='MONTH')
        elif step == 'third_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='YEAR')

        # A separate thread to populate personal histories into conversations
        elif step == 'init_conversation':
            prompt = prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.init_personal_history, period='INIT')
        elif step == 'first_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.first_expand_personal_history, period='WEEK')
        elif step == 'second_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.second_expand_personal_history, period='MONTH')
        elif step == 'third_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.third_expand_personal_history, period='YEAR')

        # Reflect on the conversation
        elif step == 'reflect_init_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.init_personal_history, 'conversation_block': data}, round=action, period='INIT')
        elif step == 'reflect_first_expand_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.first_expand_personal_history, 'conversation_block': data}, round=action, period='WEEK')
        elif step == 'reflect_second_expand_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.second_expand_personal_history, 'conversation_block': data}, round=action, period='MONTH')
        elif step == 'reflect_third_expand_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.third_expand_personal_history, 'conversation_block': data}, round=action, period='YEAR')

        elif step == 'expand_conversation_section':
            prompt = prompts.prompts_for_expanding_conversation_section(topic, data)

        elif step == 'qa_helper':
            prompt = prompts.prompts_for_generating_qa(data, action)
        elif step == 'prepare_new_content':
            prompt = prompts.prompt_for_preparing_new_content(data, action, data_type)
        elif step == 'new_content':
            prompt = prompts.prompt_for_content_generation(data, action)
        elif step == 'eval_new_content':
            prompt = prompts.prompt_for_evaluating_content(data, action)
        else:
            raise ValueError(f'Invalid step: {step}')

        # Independent API calls every time
        if step == 'expand_persona' or step == 'qa_helper' or step == 'expand_conversation_section' or step == 'translate_code' or step == 'rewrite_email' or step == 'rewrite_creative_writing':
            model = 'gpt-4o-mini' if step == 'expand_conversation_section' else self.args['models']['llm_model']
            response = self.client.chat.completions.create(
                model=model,
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
            elif step == 'reflect_conversation':
                curr_thread = self.thread_reflect_conversation
            elif step == 'prepare_new_content':
                curr_thread = self.thread_preparing_new_content
            elif step == 'new_content':
                curr_thread = self.thread_new_content
            elif step == 'eval_new_content':
                curr_thread = self.thread_eval_new_content
            elif step == 'random_question' or step == 'random_question_follow_up' or step == 'random_question_follow_up_response':
                curr_thread = self.thread_irrelevant
            else:
                curr_thread = self.thread_persona

            message = self.client.beta.threads.messages.create(
                thread_id=curr_thread.id,
                role="user",
                content=prompt
            )

            if step == 'random_question' or step == 'random_question_follow_up' or step == 'random_question_follow_up_response':
                run = self.client.beta.threads.runs.create_and_poll(
                    thread_id=curr_thread.id,
                    assistant_id=self.assistant.id
                )
            else:
                run = self.client.beta.threads.runs.create_and_poll(
                    thread_id=curr_thread.id,
                    assistant_id=self.assistant_irrelevant.id
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
                        print(f'{utils.Colors.OKGREEN}{topic}{utils.Colors.ENDC}' if topic else '')
                        print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
            else:
                response = None
                print(run.status)

        # Save general personal history to be shared across contexts
        if idx_topic == 0:
            # pattern = r'^\s*"\[(Fact|Updated Fact)\] (Likes|Dislikes)":.*$'
            # processed_response = "\n".join([line for line in response.split("\n") if not re.match(pattern, line)])
            if step == 'init_general_personal_history':
                self.general_personal_history = response
                # self.init_general_personal_history = response
            elif step == 'first_expand_general_personal_history':
                self.general_personal_history += response
                # self.first_expand_general_personal_history = response
            elif step == 'second_expand_general_personal_history':
                self.general_personal_history += response
                # self.second_expand_general_personal_history = response
            elif step == 'third_expand_general_personal_history':
                self.general_personal_history += response
                # self.third_expand_general_personal_history = response
            if step == 'expand_persona':
                self.expanded_persona = response

        # Save general+contextual personal history in order to generate conversations
        # if step == 'init_general_personal_history':
        #     self.init_personal_history = response
        if step == 'init_contextual_personal_history':
            self.init_personal_history += response
        # elif step == 'first_expand_general_personal_history':
        #     self.first_expand_personal_history = response
        elif step == 'first_expand_contextual_personal_history':
            self.first_expand_personal_history += response
        # elif step == 'second_expand_general_personal_history':
        #     self.second_expand_personal_history = response
        elif step == 'second_expand_contextual_personal_history':
            self.second_expand_personal_history += response
        # elif step == 'third_expand_general_personal_history':
        #     self.third_expand_personal_history = response
        elif step == 'third_expand_contextual_personal_history':
            self.third_expand_personal_history += response

        return response
