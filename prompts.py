import torch
import random
from utils import *


def prompts_for_background_data(content):
    prompt = "Please look at the following conversation and understand its content, format, length, and style:\n\n" + content
    return prompt


def prompts_for_expanding_persona(persona, start_time):
    birth_year = str(int(start_time.split('/')[2]) - 18)
    prompt = "The current version of the persona is short. Keep the same style and pronouns, but expand it with additional information to around five sentences. " \
             "Add a name, a gender identity, and a racial identity, if any of them is missing from the initial version." \
             "You should also include 5 personal hobbies and 5 things this person dislikes, using bullet points, all related to the persona. "\
             "List things this person may dislike, but avoid negative wording. Focus on things others might like that don’t match this person’s taste. " \
             "Adjust the persona if necessary given the person is born in " + birth_year + ". Here is the persona: " + persona
    return prompt


def prompts_for_init_general_personal_history(persona, start_time):
    prompt = "Given the following persona, expand it with 10 person's general background history within ten years starting at " + start_time + "." \
             "Turn each point into the format of a bullet point, and add a timestamp in the format of MM/DD/YYYY for each bullet point. " \
             "Remember that these events should be general like career development, and they will be shared across multiple different contexts." \
             "You should mention both daily activities and important key milestones, and both positive and negative history events. Also relate history to what this person prefers and dislikes. " \
             "Use JSON format where each timestamp is a key in the JSON dictionary. Each point should also be marked with labels of either ['Short-Term'] or ['Long-Term'], " \
             "where short-term fact refers to something happening daily, which can be irrelevant to the persona like what the person eats, " \
             "which should come with temporal quantifiers like 'today' or so, but long-term fact refers to some key personas that won't be changed for at least a year. " \
             "There should be 5 short-term and 5 long-term events. Include all 10 things this person likes and dislikes mentioned in the persona, and rewrite them as appropriate events. " \
             "Each event must come with the related personal hobbies or dislikes, marked using a key '[Fact] Likes:' or '[Fact] Dislikes:'." \
             "All events must have an appropriate time stamp in the format of MM/DD/YYYY. List at least 10 events, more are welcome. Here is the persona: " + persona
    return prompt


def prompts_for_init_contextual_personal_history(context, start_time, persona, general_personal_history):
    if general_personal_history is None:
        prompt = "Given the persona and the person's general background history above, continue to write 10 personal hobbies and 10 things this person dislikes to do but others might like, using bullet points, related to " + context + ". " \
                 "Next, write 10 more events related to the context of " + context + ". Include these 20 new things this person likes and dislikes, and rewrite them as appropriate events." \
                 "Do NOT mention anything already mentioned above. Do NOT mention anything about the general personal history, like the professional development. " \
                 "Use the same JSON format with MM/DD/YYYY timestamp starting at " + start_time + ", and use short-term/long-term labels as above. There should be 5 short-term and 5 long-term events."
    else:
        prompt = "Here is the persona:\n\n" + persona + "\n\nHere are 10 events related to the person's general background history:\n\n" + general_personal_history + "\n\n" \
                 "Given the persona and the person's general background history above, continue to write 5 personal hobbies and 5 things this person dislikes to do but others might like, using bullet points, related to " + context + ". " \
                 "Next, write 10 more events related to the context of " + context + ". Include all these 10 new things this person likes and dislikes, and rewrite them as appropriate events." \
                 "Do NOT mention anything already mentioned above. Do NOT mention anything about the general personal history, like the professional development. " \
                 "Each event must come with the related personal hobbies or dislikes, marked using a key '[Fact] Likes:' or '[Fact] Dislikes:'." \
                 "Use the same JSON format with MM/DD/YYYY timestamp from " + start_time + ", and use short-term/long-term labels as above. There should be 5 short-term and 5 long-term events."
    return prompt


def prompts_for_expanding_personal_history(context=None, type='general', period='WEEK'):
    if type != 'general':
        assert context is not None

    if type == 'general':
        prompt = "Given the initial general personal history, think about what would happen to the same person in a " + period + ". "
    else:
        prompt = "Given the initial contextual personal history, think about what would happen to the same person in a " + period + " related to the " + context + ". "
    prompt += "More than half of those new points could be, though logically still make sense, but contradictory to the original persona and personal history, especially those ['Short-Term'] facts." \
              "If there is any contradictions or knowledge updates, remember to include why, i.e., the user's reasons and intentions using an additional key '[Reasons of Change]'. Try finding interesting reasons unique to this person. " \
              "Please use the following keys, and do NOT modify the name of these keys:\n\n" \
              "key '[Old Event]' to mention the related old event contradictory to it, the key '[Old Event Date]' to mention its timestamp MM/DD/YYYY, " \
              "and the key '[Old Fact] Likes' or '[Old Fact] Dislikes' to mention the underlying like or dislike of this peron." \
              "If this is a new event without contradiction with previous ones, marked related personal hobbies or dislikes using the key '[Fact] Likes:' or '[Fact] Dislikes:', but do NOT include the '[Reasons of Change]' key.\n\n" \
              "Any contradictions should focus on what this person prefers and dislikes. " \
              "You shall also include some contradictions to the existing contradictions in the previous history, back and forth. For example, the person may like one thing, dislike it, and in some cases come back to like it again." \
              "Now, please continue to write 10 more events aligned with this persona. Do NOT repeat anything already mentioned above. " \
              "Use the same JSON format with MM/DD/YYYY timestamp starting at the end of the previous general personal history, and use short-term/long-term labels as above. There should be 5 short-term and 5 long-term events."
    return prompt


def prompts_for_generating_conversations(context, persona, curr_personal_history=None, period='INIT'):
    if context == 'therapy':
        context_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif context == 'legal':
        context_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    elif context == 'food':
        context_name, user, agent = 'food recommendation', 'Customer', 'Agent'
    else:
        raise ValueError("Invalid context", context)

    prompt = "Your task is to rewrite the following list of events related to a personal history as a format of conversation record under the context of " + context_name + ". " \
             "The conversation should strictly follow each event mentioned by the personal history and explicitly mention these events, using them and their time stamps as the skeleton. " \
             "Think about what the person's persona and history could cause trouble so that the person seeks a " + agent.lower() + ". " \
             "Write the conversation as a list in the JSON format, where each sentence is an element in the list and starts with either '" + user + "', '" + agent + "', or 'Side_Note'." \
             "Make sure to include all the bullet points in the history in the JSON file, such that there must be a separate line in square bracket '[]' that starts with 'Side_Note'" \
             "containing the related event itself and the MM/DD/YYYY timestamp before an actual sentence in the conversation that is related to this point. Do not mention underlying '[Fact]' of the event. " \
             "If a sentence is not relevant to any bullet point, no need for the 'Side_Note' before it. " \
             "The " + user.lower() + "'s conversation should clearly include detailed info about these events, while ensuring the conversation is LONG enough and contain other information and details to make it long. " \
             "If the personal history mentions about any '[Reasons of Change]', make sure to mention them naturally in the conversation and show that the person has changed the like/dislike attitude towards it, but avoid talking about the corresponding '[Old Event]' explicitly. " \

    if period != 'init':
        prompt += "Make sure to include all mentioned reasons and intentions for any changes naturally in the new conversation. "

    if period == 'INIT':
        prompt += "Here is the persona:\n\n" + persona + "\n\nand the detailed background development history:\n\n" + curr_personal_history + "\n\n"
    elif period == 'WEEK':
        prompt += "Please use the same persona:\n\n" + persona + "\n\n" \
                  "but with a new background development history happened in the next week following the previous conversation:\n\n" + curr_personal_history + "\n\n"
    elif period == 'MONTH':
        prompt += "Please use the same persona:\n\n" + persona + "\n\n" \
                  "but with a new background development history happened in the next month following the previous conversation:\n\n" + curr_personal_history + "\n\n"
    else:
        prompt += "Please use the same persona:\n\n" + persona + "\n\n" \
                  "but with a new background development history happened in the next year following the previous conversation:\n\n" + curr_personal_history + "\n\n"

    return prompt


def prompt_for_recommendations(context):
    if context == "therapy":
        user, agent = 'patient', 'therapist'
    elif context == 'legal':
        user, agent = 'client', 'lawyer assistant'
    else:
        raise ValueError("Invalid context", context)
    prompt = "Could you, as a " + agent + ", find one unique personal preference of the user different from other common " + user + "s, oriented by their persona information," \
             " and then offer a brief recommendation to the " + user + " based on this unique preference."
    return prompt