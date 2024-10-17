import torch
import random
from utils import *


def prompts_for_background_data(content):
    prompt = "Please look at the following conversation and understand its content, format, length, and style:\n\n" + content
    return prompt


def prompts_for_expanding_persona(persona):
    prompt = "The current version of the persona is short. Keep the same style and pronouns, but expand it with additional information to around five sentences: " + persona
    return prompt


def prompts_for_init_general_personal_history(persona):
    prompt = "Given the following persona, expand it with the person's general background history within ten years, " \
             "turn each point into the format of a bullet point, and add a timestamp in the format of MM/DD/YYYY for each bullet point. " \
             "Remember that these events should be general like career development, and they will be shared across multiple different contexts." \
             "You should mention both daily activities and important key milestones, and both positive and negative history events. " \
             "Use JSON format where each timestamp is a key in the JSON dictionary. Each point should also be marked with labels of either ['Short-Term'] or ['Long-Term'], " \
             "where short-term fact refers to something happening daily, which can be irrelevant to the persona like what the person eats, " \
             "which should come with temporal quantifiers like 'today' or so, but long-term fact refers to some key personas that won't be changed for at least a year. " \
             "All facts must have an appropriate time stamp in the format of MM/DD/YYYY. List at least 10 events, more are welcome. Here is the persona: " + persona
    return prompt


def prompts_for_init_contextual_personal_history(context, persona=None, init_general_personal_history=None):
    if init_general_personal_history is None:
        prompt = "Given the persona and the person's general background history above, continue to write 10 more events related to the context of " + context + ". " \
                 "Do NOT mention anything already mentioned above. Do NOT mention anything about the general personal history, like the professional development. " \
                 "Use the same JSON format with MM/DD/YYYY timestamp and short-term/long-term labels as above. "
    else:
        prompt = "Here is the persona:\n\n" + persona + "\n\nHere are 10 events related to the person's general background history:\n\n" + init_general_personal_history + "\n\n" \
                 "Given the persona and the person's general background history above, continue to write 10 more events related to the context of " + context + ". " \
                 "Do NOT mention anything already mentioned above. Do NOT mention anything about the general personal history, like the professional development. " \
                 "Use the same JSON format with MM/DD/YYYY timestamp and short-term/long-term labels as above. "
    return prompt


def prompts_for_init_therapy_conversations():
    prompt = "Your task is to write a therapy conversation record based on the persona and detailed background development history above. " \
             "Think about what the person's persona and history could cause trouble so that the person seeks a therapist. " \
             "Write the conversation as a list in the JSON format, where each sentence is an element in the list and starts with either 'Patient', 'Therapist', or 'Side_Note'." \
             "Make sure to include all the bullet points in the history in the JSON file, such that there must be a separate line in square bracket '[]' that starts with 'Side_Note'" \
             "containing the related event and the MM/DD/YYYY timestamp before an actual sentence in the conversation that is related to this point. " \
             "If a sentence is not relevant to any bullet point, no need for the 'Side_Note' before it. " \
             "The patent's conversation should clearly include detailed info about these events, while ensuring the conversation is LONG enough and contain other information and details to make it long. "
    return prompt


def prompts_for_second_general_personal_history_and_therapy_conversations(context, expanded_general_personal_history=None):
    if expanded_general_personal_history is None:
        prompt = "Write another separate therapy conversation history that is happening in a year with the same person. " \
                 "Based on the persona and personal history, what would the person do in the next week, month, and year? " \
                 "Those new points should be, though logically still make sense, but contradictory to the original persona and personal history, especially those ['Short-Term'] facts. " \
                 "Similarly, write the conversation as a list in the JSON format, where each sentence is an element in the list and starts with either 'Patient', 'Therapist', or 'Side_Note'." \
                 "Make sure to include all the bullet points in the history in the JSON file, such that there must be a separate line in square bracket '[]' that starts with 'Side_Note'" \
                 "containing the related event and the MM/DD/YYYY timestamp before an actual sentence in the conversation that is related to this point. " \
                 "If a sentence is not relevant to any bullet point, no need for the 'Side_Note' before it. " \
                 "The patent's conversation should clearly include detailed info about these events, " \
                 "Crease a JSON file. You should first list at least 10 new points in the JSON format using key 'Expanded General Personal History', more are welcome, " \
                 "and 5 more new points more related to the context of " + context + " using the key 'Expanded Contextual Personal History', " \
                 "then write the new therapy conversation using key 'Expanded Conversation' with contents as a list in JSON, same as before." \
                 "The new therapy conversation should cover all new points. It should be LONG enough and contain other information and details to make it long. "
    else:
        prompt = "Write another separate therapy conversation history that is happening in a year with the same person. " \
                 "Similarly, write the conversation as a list in the JSON format, where each sentence is an element in the list and starts with either 'Patient', 'Therapist', or 'Side_Note'." \
                 "Make sure to include all the bullet points in the history in the JSON file, such that there must be a separate line in square bracket '[]' that starts with 'Side_Note'" \
                 "containing the related event and the MM/DD/YYYY timestamp before an actual sentence in the conversation that is related to this point. " \
                 "If a sentence is not relevant to any bullet point, no need for the 'Side_Note' before it. " \
                 "The patent's conversation should clearly include detailed info about these events, " \
                 "You should first incorporate the following expanded general personal history happening in the past year:\n\n" + expanded_general_personal_history + "\n\n" \
                 "Next, crease a JSON file, list 5 more new points more related to the context of " + context + " using the key 'Expanded Contextual Personal History', " \
                 "then write the new therapy conversation using key 'Expanded Conversation' with contents as a list in JSON, same as before." \
                 "The new therapy conversation should cover all new points. It should be LONG enough and contain other information and details to make it long. "
    return prompt


def prompt_for_question_answer_pairs():
    prompt = "Generate 10 question-answer pairs directly related to the two JSON files above. " \
             "These questions should ask about obvious key facts related to the description of each event, but avoid asking questions related to their labels or timestamps. " \
             "Each question should be able to be answered by only looking at the conversations without looking at the side notes in the records. " \
             "All question-answer pairs should be objective and specific. Write your question-answer pairs in the JSON format. " \
             "Each element should have an unique integer index as the key, starting from '0' and incrementing by one at a time, " \
             "and each value should be another dict with keys 'Question', 'Answer', and 'Reference', " \
             "where 'Reference' should mention the related detailed personal history in the JSON files above including the timestamp in the format of MM/DD/YYYY. " \
             "Clearly mark if this question-answer pair is related to one, two, or more events, " \
             "and whether it is related to the general personal history ('General') or the contextual personal history ('Contextual')."
    return prompt
