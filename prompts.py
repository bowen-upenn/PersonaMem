import torch
import random
from utils import *


def prompts_for_background_data(content):
    prompt = "Please look at the following conversation and understand its content, format, length, and style:\n\n" + content
    return prompt


def prompts_for_init_general_bullet_points(persona):
    prompt = "Given the following persona, expand it with the person's general background development history within ten years, " \
             "turn each point into the format of a bullet point, and add a timestamp in the format of MM/DD/YYYY for each bullet point. " \
             "You should mention both daily activities and important key milestones, and both positive and negative history events. " \
             "Use JSON format where each timestamp is a key in the JSON dictionary. Each point should also be marked with labels of either ['Short-Term'] or ['Long-Term'], " \
             "where short-term fact refers to something happening daily, which can be irrelevant to the persona like what the person eats, " \
             "which should come with temporal quantifiers like 'today' or so, but long-term fact refers to some key personas that won't be changed for at least a year. " \
             "Here is the persona: " + persona
    return prompt


def prompts_for_init_contextual_bullet_points(persona, general_bullet_points):
    prompt = ""
    return prompt


def prompts_for_init_therapy_conversations():
    prompt = "Your task is to write a therapy conversation record based on the persona and detailed background development history above. " \
             "Think about what the person's persona and history could cause trouble so that the person seeks a therapist. " \
             "Make sure to include all the bullet points in the history in the JSON file. " \
             "Write the conversation as a list in the JSON format, where each sentence is an element in the list and starts with either 'Patient', 'Therapist', or 'Side_Note'." \
             "If there is a sentence in the patient's conversation that is related to a bullet point, " \
             "add an separate line in square bracket '[]' that starts with 'Side_Note' immediately after that sentence in the list, which includes the related event and the MM/DD/YYYY timestamp. " \
             "The patent's conversation should clearly include detailed info about these events, " \
             "while ensuring the conversation is LONG enough and contain other information and details to make it long. "
    return prompt


def prompts_for_second_general_bullet_points_and_therapy_conversations():
    prompt = "Write another separate therapy conversation history that is happening in a year with the same person. " \
             "Based on the persona and personal history, what would the person do in the next week, month, and year? " \
             "Those new points should be, though logically still make sense, but contradictory to the original persona and personal history, especially those ['Short-Term'] facts. " \
             "Similarly, write the conversation as a list in the JSON format, where each sentence is an element in the list and starts with either 'Patient', 'Therapist', or 'Side_Note'." \
             "If there is a sentence in the patient's conversation that is related to a bullet point, " \
             "add an separate line in square bracket '[]' that starts with 'Side_Note' immediately after that sentence in the list, which includes the related event and the MM/DD/YYYY timestamp. " \
             "The patent's conversation should clearly include detailed info about these events, " \
             "Crease a JSON file. You should list these new points first in the JSON format using key 'Expanded Personal History', " \
             "and then write the new therapy conversation using key 'Expanded Conversation' with contents purely in a textual format."
    return prompt


def prompt_for_question_answer_pairs():
    prompt = "Generate 10 question-answer pairs directly related to the two JSON files above. " \
             "These questions should ask about obvious key facts related to the description of each event, but avoid asking questions related to their labels or timestamps. " \
             "Each question should be able to be answered by only looking at the conversations without looking at the side notes in the records. " \
             "All answers should be objective. Write your question-answer pairs in the JSON format. " \
             "Each element should have an unique integer index as the key, starting from '0' and incrementing by one at a time, " \
             "and each value should be another dict with keys 'Question', 'Answer', and 'Reference', " \
             "where 'Reference' should mention the related detailed personal history in the JSON files above including the timestamp in the format of MM/DD/YYYY. " \
             "Clearly mark if this question-answer pair is related to one, two, or more events."
    return prompt