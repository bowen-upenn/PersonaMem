import torch
import random
from pydantic import BaseModel


def prompts_for_background_data(content):
    prompt = "Please look at the following conversation and understand its content, format, length, and style:\n\n" + content
    return prompt


def prompts_for_random_question_follow_up():
    prompt = "Find a follow-up question based on the previous question and response. Output the question only. No other words."
    return prompt


def prompts_for_elaborating_topic(topic):
    prompt = "Please elaborate what a person would like to talk about under the topic of " + topic + ". "
    return prompt


def prompts_for_expanding_persona(persona, start_time):
    birth_year = str(int(start_time.split('/')[2]) - 18)
    gender_identity = (' transgender ' if random.random() < 0.2 else '') + random.choice(['female', 'male', 'non-binary'])
    racial_identity = random.choice(['Asian', 'South Asian', 'African American', 'Hispanic', 'Indigenous', 'White', 'Jewish', 'Pacific Islander', 'Mixed race'])
    prompt = "The current version of the persona is short. Keep the same style and pronouns, but expand it with additional information to around five sentences. " \
             "Add a name, a gender identity of " + gender_identity + ", and a racial identity of " + racial_identity + ", if any of them is missing from the initial version." \
             "Adjust the persona if necessary given the person is born in " + birth_year + ". Here is the persona: " + persona
    return prompt


def prompts_for_init_general_personal_history(persona, start_time):
    prompt = "Given the following persona, expand it with 10 person's general background history within ten years starting at " + start_time + "." \
             "Turn each point into the format of a bullet point, and add a timestamp in the format of MM/DD/YYYY for each bullet point. " \
             "Remember that these events should be general like career development, and they will be shared across multiple different topics." \
             "You should mention both daily activities and important key milestones, and both positive and negative history events. Also relate history to what this person prefers and dislikes. " \
             "Use JSON format where each timestamp is a key in the JSON dictionary. Each point should also be marked with labels of either ['Short-Term'] or ['Long-Term'], " \
             "where short-term fact refers to something happening daily, which can be irrelevant to the persona like what the person eats, " \
             "which should come with temporal quantifiers like 'today' or so, but long-term fact refers to some key personas that won't be changed for at least a year. " \
             "There should be 5 short-term and 5 long-term events. Include all 10 things this person likes and dislikes mentioned in the persona, and rewrite them as appropriate events. " \
             "All events must have an appropriate time stamp in the format of MM/DD/YYYY. List at least 10 events, more are welcome. " \
             "Here is the template you should follow for each event:\n\n" \
                '"MM/DD/YYYY": {\n' \
                    '"Event": xxx, \n' \
                    '"Category": "Short-Term" OR "Long-Term"\n' \
                "}, \n\n" \
             "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'MM/DD/YYYY' and 'xxx' in the template. Please use DOUBLE quotes in order to generate the correct JSON format." \
             "Here is the persona: " + persona
    # schema = {
    #     "type": "object",
    #     "patternProperties": {
    #         "^\\d{2}/\\d{2}/\\d{4}$": {
    #             "type": "object",
    #             "properties": {
    #                 "Event": {"type": "string"},
    #                 "Category": {"type": "string", "enum": ["Short-Term", "Long-Term"]}
    #             },
    #             "required": ["Event", "Category"],
    #             "additionalProperties": False
    #         }
    #     },
    #     "minProperties": 10,
    #     "additionalProperties": False
    # }

    return prompt


def prompts_for_init_contextual_personal_history(topic, start_time, persona, general_personal_history):
    prompt = "Here is the persona:\n\n" + persona + "\n\nHere are some events related to the person's general background history:\n\n" + general_personal_history + "\n\n" \
             "Given the persona above, please first list 20 hobbies related to " + topic + ". Next, please randomly assign 10 of them to the likes of this person, and remaining 10 to the dislikes of this person. " \
             "Make sure every hobby, regardless of whether it is a like or dislike, is unique and attractive in common, so that the exact dislikes can potentially be turned into likes in the future." \
             "please list 10 unique personal hobbies and 10 things this person dislikes but others might still like, using bullet points, related to " + topic + ". " \
             "Next, write 10 more events related to the topic of " + topic + ". Think about how this person's general background history may affect their events under " + topic + \
             "Include all these 20 new things this person likes and dislikes, and rewrite them as appropriate events." \
             "Do NOT mention anything already mentioned above. Do NOT mention anything about the general personal history, like the professional development. " \
             "Each event must come with the related personal hobbies or dislikes, marked using a key '[Fact] Likes:' or '[Fact] Dislikes:' closely associated with the 20 things you listed here, and they should concentrate on the topic of " + topic + \
             "If an event is related to a dislike, it should show that this person dislikes it after experienced it or the person is trying to avoid it. " \
             "Use the same JSON format with MM/DD/YYYY timestamp from " + start_time + ", and use short-term/long-term labels as above. There should be 10 short-term and 10 long-term events." \
             "List all 20 hobbies first, and then randomly assign those 20 hobbies into likes or dislikes for this person. " \
             "After you have generated the list above, generate one dict for each event following those 20 likes and dislikes. " \
             "List all 20 hobbies first, and then follow this template in string to randomly assign those 20 hobbies into likes or dislikes for this person:\n\n" \
             "20 hobbies: xxx, ..., xxx\n" \
             "Initial preferences randomly assigned: [1] Likes xxx\n" \
             "[2] Likes xxx\n" \
             "[3] Likes xxx\n" \
             "[4] Likes xxx\n" \
             "[5] Likes xxx\n" \
             "[6] Likes xxx\n" \
             "[7] Likes xxx\n" \
             "[8] Likes xxx\n" \
             "[9] Likes xxx\n" \
             "[10] Likes xxx\n" \
             "[1] Dislikes xxx\n" \
             "[2] Dislikes xxx\n" \
             "[3] Dislikes xxx\n" \
             "[4] Dislikes xxx\n" \
             "[5] Dislikes xxx\n" \
             "[6] Dislikes xxx\n" \
             "[7] Dislikes xxx\n" \
             "[8] Dislikes xxx\n" \
             "[9] Dislikes xxx\n" \
             "[10] Dislikes xxx\n" \
             "After you have generated the list above, here is the template in JSON you should follow for each event. PLEASE MUST USE JSON FOR THIS PART:\n\n" \
                '"MM/DD/YYYY": {\n' \
                    '"Event": xxx, \n' \
                    '"Category": "Short-Term" OR "Long-Term"\n' \
                    '"[Fact] Likes" OR "[Fact] Dislikes": xxx, \n' \
                "}, \n\n" \
             "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'MM/DD/YYYY' and 'xxx' in the template. Please use DOUBLE quotes in order to generate the correct JSON format."

    # schema = {
    #     "title": "ContextualPersonalHistory",
    #     "description": "Structured output for generating a persona's hobbies, preferences, and related events.",
    #     "type": "object",
    #     "properties": {
    #         "Hobbies": {
    #             "type": "array",
    #             "minItems": 20,
    #             "maxItems": 20,
    #             "items": {
    #                 "type": "string"
    #             },
    #             "description": "List of 20 hobbies related to the topic."
    #         },
    #         "Preferences": {
    #             "type": "object",
    #             "properties": {
    #                 "Likes": {
    #                     "type": "array",
    #                     "minItems": 10,
    #                     "maxItems": 10,
    #                     "items": {
    #                         "type": "string"
    #                     },
    #                     "description": "10 hobbies that the persona likes, randomly selected from the 20 hobbies."
    #                 },
    #                 "Dislikes": {
    #                     "type": "array",
    #                     "minItems": 10,
    #                     "maxItems": 10,
    #                     "items": {
    #                         "type": "string"
    #                     },
    #                     "description": "10 hobbies that the persona dislikes, randomly selected from the 20 hobbies."
    #                 }
    #             },
    #             "required": ["Likes", "Dislikes"],
    #             "description": "Randomly assigned preferences for the persona."
    #         },
    #         "Events": {
    #             "type": "object",
    #             "minProperties": 20,
    #             "patternProperties": {
    #                 "^\\d{2}/\\d{2}/\\d{4}$": {
    #                     "type": "object",
    #                     "properties": {
    #                         "Event": {
    #                             "type": "string",
    #                             "description": "A short description of the event."
    #                         },
    #                         "Category": {
    #                             "type": "string",
    #                             "enum": ["Short-Term", "Long-Term"],
    #                             "description": "Indicates whether the event is a short-term occurrence or a long-term milestone."
    #                         },
    #                         "Fact": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "Type": {
    #                                     "type": "string",
    #                                     "enum": ["[Fact] Likes", "[Fact] Dislikes"],
    #                                     "description": "Indicates whether the event relates to a like or dislike."
    #                                 },
    #                                 "Value": {
    #                                     "type": "string",
    #                                     "description": "The specific hobby or dislike associated with the event."
    #                                 }
    #                             },
    #                             "required": ["Type", "Value"],
    #                             "description": "Fact must be either a liked or disliked hobby."
    #                         }
    #                     },
    #                     "required": ["Event", "Category", "Fact"],
    #                     "additionalProperties": false
    #                 }
    #             },
    #             "description": "At least 20 events related to the topic and persona preferences."
    #         }
    #     },
    #     "required": ["Hobbies", "Preferences", "Events"],
    #     "additionalProperties": false
    # }

    return prompt


def prompts_for_expanding_personal_history(topic=None, type='general', period='WEEK'):
    if type != 'general':
        assert topic is not None

    if type == 'general':
        prompt = "Given the initial general personal history, think about what would happen to the same person in a " + period + ". "
    else:
        prompt = "Given the initial contextual personal history, think about what would happen to the same person in a " + period + " related to the " + topic + ". "
    prompt += "More than half of those new points could be, though logically still make sense, but contradictory to the original persona and personal history, especially those ['Short-Term'] facts." \
              "If there is any contradictions or knowledge updates, remember to include why, i.e., the user's reasons and intentions using an additional key '[Reasons of Change]'. " \
              "Try finding some very unique and personal reasons for this person, uncommon for the general public, that trigger the change. " \
              "Please also use the following keys, and do NOT modify the name of these keys:\n\n" \
              "The key '[Old Event]' to mention the related old event contradictory to it, the key '[Old Event Date]' to mention its timestamp MM/DD/YYYY, " \
              "If this is a new event without contradiction with previous ones, marked related personal hobbies or dislikes using the key '[Fact] Likes:' or '[Fact] Dislikes:', but do NOT include the '[Reasons of Change]' key.\n\n"
    if type == 'contextual':
        prompt += "the key '[Old Fact] Likes' or '[Old Fact] Dislikes' to mention the previous like or dislike of this peron related to this event." \
                  "the key '[Updated Fact] Likes' or '[Updated Fact] Dislikes' should be exactly the OPPOSITE to its corresponding '[Old Fact] Likes' or '[Old Fact] Dislikes'."
    prompt += "Any contradictions should focus on what this person prefers and dislikes. " \
              "You shall also include some contradictions to the existing contradictions in the previous history, back and forth. For example, the person may like one thing, dislike it, and in some cases come back to like it again." \
              "Now, please continue to write 10 more events aligned with this persona. Do NOT repeat anything already mentioned above. " \
              "Use the same JSON format with MM/DD/YYYY timestamp starting at the end of the previous general personal history, and use short-term/long-term labels as above. There should be 5 short-term and 5 long-term events." \

    if type == 'general':
        prompt += "Here is the template you should follow for each event WITHOUT knowledge updates:\n\n" \
                  '"MM/DD/YYYY": {\n' \
                  '"Event": xxx, \n' \
                  '"Category": "Short-Term" OR "Long-Term"\n' \
                  "}, \n\n" \
                  "Here is the template you should follow for each event WITH knowledge updates:\n\n" \
                  "'MM/DD/YYYY': {\n" \
                      '"Event": xxx, \n' \
                      '"Category": "Short-Term" OR "Long-Term"\n' \
                      '"[Reasons of Change]": xxx, (Please find some unique, uncommon, and personal reasons!) \n' \
                      '"[Old Event Date]": MM/DD/YYYY, \n' \
                      '"[Old Event]": xxx, \n' \
                  "}\n" \
                  "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'MM/DD/YYYY' and 'xxx' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    else:
        prompt += "Here is the template you should follow for each event WITHOUT knowledge updates:\n\n" \
                  '"MM/DD/YYYY": {\n' \
                      '"[Fact] Likes" OR "[Fact] Dislikes": xxx, \n' \
                      '"Event": xxx, \n' \
                      '"Category": "Short-Term" OR "Long-Term"\n' \
                  "}, \n\n" \
                  "Here is the template you should follow for each event WITH knowledge updates:\n\n" \
                  "'MM/DD/YYYY': {\n" \
                  '"[Old Fact] Likes" OR "[Old Fact] Dislikes": xxx, \n' \
                  '"[Old Event Date]": MM/DD/YYYY, \n' \
                  '"[Old Event]": xxx, \n' \
                  '"[Updated Fact] Likes" OR "[Updated Fact] Dislikes": xxx, \n' \
                  '"[Reasons of Change]": xxx, (Please find some unique, uncommon, and personal reasons!) \n' \
                  '"Event": xxx, \n' \
                  '"Category": "Short-Term" OR "Long-Term"\n' \
                  "}\n" \
                  "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'MM/DD/YYYY' and 'xxx' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."

    # if type == 'general':
    #     schema = {
    #         "title": "GeneralPersonalHistoryExpansion",
    #         "description": "Schema for expanding general personal history, including new events and knowledge updates.",
    #         "type": "object",
    #         "minProperties": 10,
    #         "patternProperties": {
    #             "^\\d{2}/\\d{2}/\\d{4}$": {
    #                 "type": "object",
    #                 "properties": {
    #                     "Event": {
    #                         "type": "string",
    #                         "description": "A description of the new event."
    #                     },
    #                     "Category": {
    #                         "type": "string",
    #                         "enum": ["Short-Term", "Long-Term"],
    #                         "description": "Specifies whether the event is short-term or long-term."
    #                     },
    #                     "[Reasons of Change]": {
    #                         "type": "string",
    #                         "description": "A unique, uncommon, and personal reason for the contradiction or update.",
    #                         "nullable": true
    #                     },
    #                     "[Old Event Date]": {
    #                         "type": "string",
    #                         "pattern": "^\\d{2}/\\d{2}/\\d{4}$",
    #                         "description": "The date of the previous contradictory event.",
    #                         "nullable": true
    #                     },
    #                     "[Old Event]": {
    #                         "type": "string",
    #                         "description": "The old event that contradicts the new one.",
    #                         "nullable": true
    #                     }
    #                 },
    #                 "required": ["Event", "Category"],
    #                 "oneOf": [
    #                     {
    #                         "required": ["[Reasons of Change]", "[Old Event Date]", "[Old Event]"]
    #                     },
    #                     {
    #                         "not": {"required": ["[Reasons of Change]", "[Old Event Date]", "[Old Event]"]}
    #                     }
    #                 ],
    #                 "additionalProperties": false
    #             }
    #         },
    #         "additionalProperties": false
    #     }
    # else:
    #     schema = {
    #         "title": "ContextualPersonalHistoryExpansion",
    #         "description": "Schema for expanding contextual personal history, including contradictions in preferences.",
    #         "type": "object",
    #         "minProperties": 10,
    #         "patternProperties": {
    #             "^\\d{2}/\\d{2}/\\d{4}$": {
    #                 "type": "object",
    #                 "properties": {
    #                     "Event": {
    #                         "type": "string",
    #                         "description": "A description of the new event."
    #                     },
    #                     "Category": {
    #                         "type": "string",
    #                         "enum": ["Short-Term", "Long-Term"],
    #                         "description": "Specifies whether the event is short-term or long-term."
    #                     },
    #                     "[Fact] Likes": {
    #                         "type": "string",
    #                         "description": "A hobby or preference associated with the event.",
    #                         "nullable": true
    #                     },
    #                     "[Fact] Dislikes": {
    #                         "type": "string",
    #                         "description": "A dislike related to the event.",
    #                         "nullable": true
    #                     },
    #                     "[Old Fact] Likes": {
    #                         "type": "string",
    #                         "description": "Previous hobby that is now disliked.",
    #                         "nullable": true
    #                     },
    #                     "[Old Fact] Dislikes": {
    #                         "type": "string",
    #                         "description": "Previous dislike that is now liked.",
    #                         "nullable": true
    #                     },
    #                     "[Updated Fact] Likes": {
    #                         "type": "string",
    #                         "description": "A new preference that was previously disliked.",
    #                         "nullable": true
    #                     },
    #                     "[Updated Fact] Dislikes": {
    #                         "type": "string",
    #                         "description": "A new dislike that was previously liked.",
    #                         "nullable": true
    #                     },
    #                     "[Reasons of Change]": {
    #                         "type": "string",
    #                         "description": "A unique, uncommon, and personal reason for changing preferences.",
    #                         "nullable": true
    #                     },
    #                     "[Old Event Date]": {
    #                         "type": "string",
    #                         "pattern": "^\\d{2}/\\d{2}/\\d{4}$",
    #                         "description": "The date of the previous contradictory event.",
    #                         "nullable": true
    #                     },
    #                     "[Old Event]": {
    #                         "type": "string",
    #                         "description": "The old event that contradicts the new one.",
    #                         "nullable": true
    #                     }
    #                 },
    #                 "required": ["Event", "Category"],
    #                 "oneOf": [
    #                     {
    #                         "required": ["[Fact] Likes"],
    #                         "not": {"required": ["[Fact] Dislikes"]}
    #                     },
    #                     {
    #                         "required": ["[Fact] Dislikes"],
    #                         "not": {"required": ["[Fact] Likes"]}
    #                     },
    #                     {
    #                         "required": [
    #                             "[Old Fact] Likes",
    #                             "[Updated Fact] Dislikes",
    #                             "[Reasons of Change]",
    #                             "[Old Event Date]",
    #                             "[Old Event]"
    #                         ],
    #                         "not": {
    #                             "required": ["[Old Fact] Dislikes", "[Updated Fact] Likes"]
    #                         }
    #                     },
    #                     {
    #                         "required": [
    #                             "[Old Fact] Dislikes",
    #                             "[Updated Fact] Likes",
    #                             "[Reasons of Change]",
    #                             "[Old Event Date]",
    #                             "[Old Event]"
    #                         ],
    #                         "not": {
    #                             "required": ["[Old Fact] Likes", "[Updated Fact] Dislikes"]
    #                         }
    #                     }
    #                 ],
    #                 "additionalProperties": false
    #             }
    #         },
    #         "additionalProperties": false
    #     }
    return prompt


def prompts_for_generating_conversations(topic, persona, curr_personal_history=None, period='INIT'):
    if topic == 'therapy':
        topic_name, user, agent = 'therapy', 'Patient', 'Therapist'
    else:
        topic_name, user, agent = topic, 'User', 'Assistant'

    prompt = "Your task is to rewrite the following list of events related to a personal history as a format of conversation record under the topic of " + topic_name + ". " \
             "The conversation should strictly follow each event mentioned by the personal history and explicitly mention these events one by one, using them and their time stamps of the format MM/DD/YYYY as the skeleton. Do NOT change the time stamps. " \
             "Think about what the person's persona and history could cause trouble so that the person seeks a " + agent.lower() + ". " \
             "Write the conversation as a list of string, where each sentence is an element in the list and starts with either '" + user + "', '" + agent + "', or 'Side_Note'." \
             "Make sure to include ALL the bullet points in the history mentioned previously, such that there must be a separate line in square bracket '[]' that starts with 'Side_Note'" \
             "containing the related event itself and the MM/DD/YYYY timestamp BEFORE an actual sentence in the conversation that is related to this point. Do not mention underlying '[Fact]' of the event. " \
             "Do NOT modify any MM/DD/YYYY above. If a sentence is not relevant to any bullet point, no need for the 'Side_Note' before it. " \
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

    prompt += "Except for the initial sentences as the introduction, here is the template you should follow for each pair of utterance that mentions a fact in the personal history:\n\n" \
              "[\n" \
              '"Side_Note: [xxx] MM/DD/YYYY",' \
              '"' + user + ': yyy",' \
              '"' + agent + ': zzz",' \
              "...] Use a Python list of strings where each sentence is one string. Fill in the actual data at placeholders 'MM/DD/YYYY', 'xxx', 'yyy', and 'zzz' in the template. " \
              "Use double quotes for each sentence. Do NOT use JSON. Please make sure every timestamp in the Side_Note can be found in the given list of personal history. No other words."

    # schema = {
    #     "title": "GeneratedConversation",
    #     "description": f"Schema for generating structured conversations between {user} and {agent} based on personal history events.",
    #     "type": "array",
    #     "items": {
    #         "type": "string",
    #         "pattern": f"^(Side_Note: \\[.+\\] \\d{{2}}/\\d{{2}}/\\d{{4}}|{user}: .+|{agent}: .+)$",
    #         "description": f"Each entry is a string representing a conversation line between {user} and {agent}."
    #     },
    #     "minItems": 10
    # }
    return prompt


def prompts_for_reflecting_conversations(topic, data, round, period='INIT'):
    if topic == 'therapy':
        topic_name, user, agent = 'therapy', 'Patient', 'Therapist'
    else:
        topic_name, user, agent = topic, 'User', 'Assistant'

    if period == 'INIT':
        history_block = "'Init Contextual Personal History'"
        conversation_block = "'Init Conversation'"
    elif period == 'WEEK':
        history_block = "'Contextual Personal History Next Week'"
        conversation_block = "'Conversation Next Week'"
    elif period == 'MONTH':
        history_block = "'Contextual Personal History Next Month'"
        conversation_block = "'Conversation Next Month'"
    else:
        history_block = "'Contextual Personal History Next Year'"
        conversation_block = "'Conversation Next Year'"

    if round == 1:
        prompt = "Given the following " + history_block + " and the " + conversation_block + ", check if the " + conversation_block + " has covered every single timestamp in the " + history_block + ". All [Old Event Date] does NOT count! Ignore them! " \
                 "List all missed ones in the conversation, as well as those in the conversation but not in the " + history_block + ":\n\n" + history_block + "\n\n" + data['history_block'] + "\n\n" + conversation_block + "\n\n" + data['conversation_block']
        schema = None
    elif round == 2:
        prompt = "Please fill in these missed timestamps with their corresponding events mentioned in the " + history_block + " into the " + conversation_block + ". " \
                 "Make sure every single timestamp in the Side_Note in this conversation can be found in the given " + history_block + ", instead of personal history in other time periods. " \
                 "You may add some transition sentences to make it smooth, but do NOT modify any other words in the original conversation, except for the sentences with incorrect timestamps. " \
                 "If there is no missed timestamp, no need to change any part of the original conversation. " \
                 "Follow exactly the SAME template in the original conversation:\n\n" \
                 "[\n" \
                 '"Side_Note: [xxx] MM/DD/YYYY",' \
                 '"' + user + ': yyy",' \
                 '"' + agent + ': zzz",' \
                 "...] " \
                 "The whole output should be a Python list of strings where each utterance is one string. Fill in the actual data at placeholders 'MM/DD/YYYY', 'xxx', 'yyy', and 'zzz' in the template. Use double quotes for each sentence. Do NOT use JSON. Just output the completed conversation. No other words."

        # schema = {
        #     "title": "FixedConversationWithTimestamps",
        #     "description": "Schema for filling in missing or incorrect timestamps in a conversation while keeping the structure unchanged.",
        #     "type": "array",
        #     "items": {
        #         "type": "string",
        #         "pattern": f"^(Side_Note: \\[.+\\] \\d{{2}}/\\d{{2}}/\\d{{4}}|{user}: .+|{agent}: .+)$",
        #         "description": "Each entry is a string representing a conversation line with corrected timestamps."
        #     },
        #     "minItems": 10
        # }
    else:
        raise ValueError("Invalid round", round)
    return prompt


def prompts_for_expanding_conversation_section(topic, data):
    if topic == 'therapy':
        topic_name, user, agent = 'therapy', 'Patient', 'Therapist'
    else:
        topic_name, user, agent = topic, 'User', 'Assistant'

    prompt = "Please expand these sentences. I do NOT want any new user preferences, examples, or changes to the story behind the conversation. " \
             "Instead, extend each line to AT LEAST FIVE sentences by adding additional details or irrelevant topic that delves deeper into the mentioned objects or events. " \
             "Ensure that no new preferences are introduced or altered. Each revised sentence should provide greater depth while maintaining consistency with the original narrative and intent." \
             "Note that the lines said by " + agent + " should be even longer to show the caring or professionalism. " \
             "Also note that if the last line is another line of 'Side_Note', that 'Side_Note' indicates the next event, so the previous line should consider how to smoothly transit the conversation. " \
             "Here is the section you should expand, while do NOT expand or modify the line(s) of Side_Note.\n\n" + '\n'.join(data['section']) + "\n\n" \
             "Please remove or rephrase any timestamp MM/DD/YYYY mentioned by the " + user + " and " + agent + " in their utterances. Note that this conversation is happening at " + data['last_timestamp'] + "." \
             "But you should keep the Side_Note unmodified. Each Side_Note should include the original timestamp MM/DD/YYYY. " \
             "Follow the template in the original sentences:\n\n" \
             "[\n" \
             '"Side_Note: [xxx] MM/DD/YYYY" (Please include MM/DD/YYYY here),' \
             '"' + user + ': yyy" (More than 5 sentences. Do NOT include MM/DD/YYYY here),' \
             '"' + agent + ': zzz"  (More than 10 sentences. Do NOT include MM/DD/YYYY here),' \
             "...]\n Use a Python list of strings where each sentence is one string. Fill in the actual data at placeholders 'MM/DD/YYYY', 'xxx', 'yyy', and 'zzz' in the template. Use double quotes for each sentence. " \
             "The actual order of " + user + " and " + agent + " in your expanded sentences should follow the original order in the original sentences. Please output only a valid Python list of strings in a code block with no extra text."

    # schema = {
    #     "title": "ExpandedConversationSection",
    #     "description": f"Schema for expanding conversation lines between {user} and {agent} while maintaining logical consistency and structure.",
    #     "type": "array",
    #     "items": {
    #         "type": "string",
    #         "pattern": f"^(Side_Note: \\[.+\\] \\d{{2}}/\\d{{2}}/\\d{{4}}|{user}: (?:[^.!?]+[.!?]\\s*){{5,}}|{agent}: (?:[^.!?]+[.!?]\\s*){{10,}})$",
    #         "description": f"Each entry is a string representing an expanded conversation line between {user} and {agent}, requiring at least 5 sentences for {user} and 10 sentences for {agent}."
    #     },
    #     "minItems": 10
    # }

    return prompt


def prompts_for_generating_qa(data, action):
    if action == 'recall_facts':
        prompt = "We want to evaluate whether a chatbot can remember factual information (NOT the user's preferences toward it) shared by the user during previous conversations, " \
                 "and whether the model can utilize its memory to provide a personalized response. Given this specific activity\n\n'" + data['related_fact'] + "'\n\ndescribed by the user in a conversation with the chatbot:\n\n" + data['user_utterance'] + "\n\n" \
                 "What question might the user query the chatbot model to bring up this topic again? Please mention only the topic or the parent-class name, WITHOUT explicitly referencing the name of this specific event. " \
                 "Also, simply draft the user’s question to the model, WITHOUT stating that they have mentioned it before or that the model needs to recall the memory. " \
                 "Make the user question more detailed with some topic. Remember that the user is asking this question to an LLM, not a real human. " \
                 "Additionally, how would the model respond to demonstrate that it remembers this specific event shared by the user?" \
                 "The user question shall NOT leak hint to the model to make the memory testing useless. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Question": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "ChatbotMemoryRecallEvaluation",
        #   "description": "Schema for evaluating a chatbot’s ability to recall factual information shared by the user in prior conversations.",
        #   "type": "object",
        #   "properties": {
        #     "User Question": {
        #       "type": "string",
        #       "description": "A detailed question from the user that indirectly brings up a previously mentioned factual topic without explicitly referencing the exact event."
        #     },
        #     "Model Response": {
        #       "type": "string",
        #       "description": "A response demonstrating that the chatbot remembers and correctly references the previously shared factual information."
        #     }
        #   },
        #   "required": ["User Question", "Model Response"],
        #   "additionalProperties": false
        # }
    elif action == 'propose_incorrect_facts':
        prompt = "This is the correct personalized response to the question: " + data['question'] + ": " + data['response'] + "\n\n" \
                 "Please propose three incorrect options to prepare a multiple choice Q&A, keeping all incorrect responses generally good but mentioning different things or activities. " \
                 "Each option should share similar tone, matching length, and equal level of detail. Please do NOT be lazy! " \
                 "Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization." \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'
        # schema = {
        #   "title": "IncorrectFactsProposals",
        #   "description": "A list of three incorrect factual responses for multiple-choice Q&A, ensuring each option is well-written and equal in length to the correct response.",
        #   "type": "array",
        #   "items": {
        #     "type": "string",
        #     "description": "An incorrect response option with similar tone, detail, and length to the correct response."
        #   },
        #   "minItems": 3,
        #   "maxItems": 3
        # }
    elif action == 'recall_facts_inverse':
        prompt = "We want to evaluate whether a chatbot can remember factual information (NOT the user's preferences toward it) shared by the user during previous conversations, " \
                 "and whether the model can utilize its memory to provide a personalized response. Given this specific activity described by the user in a conversation with the chatbot:\n\n" + data['event'] + "\n\n" \
                 "What question might the user ask the chatbot to bring up this topic again? Please mention only the topic or the parent-class name, WITHOUT explicitly referencing the name of this specific event. " \
                 "Also, simply mimic the user’s question, WITHOUT stating that they have mentioned it before or that the model needs to recall the memory." \
                 "Most importantly, the user should say they want to try something new, WITHOUT explicitly saying what they have done before to test the model's memory capability. " \
                 "Make the user question more detailed with some topic. Remember that the user is asking this question to an LLM, not a real human. " \
                 "Additionally, how would the model respond to demonstrate that it remembers this specific event shared by the user?" \
                 "The model's response should simply give an answer, WITHOUT first mentioning what the user has already done before. " \
                 "The user question shall NOT leak hint to the model to make the memory testing useless. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Question": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'xxx' and 'yyy' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "RecallFactsInverse",
        #   "description": "Schema for testing chatbot memory recall when the user asks about trying something new, indirectly referencing a past event.",
        #   "type": "object",
        #   "properties": {
        #     "User Question": {
        #       "type": "string",
        #       "description": "A detailed question from the user that hints at a previously mentioned topic but does not explicitly mention past experiences."
        #     },
        #     "Model Response": {
        #       "type": "string",
        #       "description": "A response demonstrating that the chatbot remembers and correctly references the previously shared factual information without stating what the user has done before."
        #     }
        #   },
        #   "required": ["User Question", "Model Response"],
        #   "additionalProperties": false
        # }
    elif action == 'propose_incorrect_facts_inverse':
        prompt = "Given this question from the user: " + data['question'] + ", please create three responses inspired by these conversations from other users. " \
                 "Since they originate from other users, it is safe to use them here.\n\n" + data['random_event_histories'] + "\n\n" \
                 "Each option should share similar tone, matching length, and equal level of detail. Please do NOT be lazy! " \
                 "Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization." \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Fill in the actual data at placeholders "xxx" and "yyy" in the template. Please use double quotes for each string. No other words.'
        # schema = {
        #   "title": "IncorrectFactsInverseProposals",
        #   "description": "A list of three incorrect responses inspired by other user conversations, ensuring each option is well-written and equal in length to the correct response.",
        #   "type": "array",
        #   "items": {
        #     "type": "string",
        #     "description": "An incorrect response option that matches the tone, detail, and length of the correct response."
        #   },
        #   "minItems": 3,
        #   "maxItems": 3
        # }

    elif action == 'generalize_reason_to_other_scenarios':
        prompt = "The user has mentioned the detailed reason below of their preference update in previous conversations:\n\n" + data['event'] + "\n\n" \
                 "You should focus on the [Reasons of Change] part. We actually want to evaluate if the model can remember and utilize this reason of change as a motivation to this user, " \
                 "and then generalize the reason to other scenarios the same user might say in the near future during the conversation, not the event or activity itself. " \
                 "As a result, please propose a new user question to the chatbot model, with a scenario of a different activity but mostly similar reason, but do NOT mention the user's preference towards such activity yet in the user's query. " \
                 "Remember that the user is asking this question to an LLM, not a real human. " \
                 "Please also propose a model's response to assume the user's preference based on this reason. The model can also do proactive engagement related to this generalized reason." \
                 "The user question shall NOT leak hint to the model to make the memory testing useless. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Question": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'xxx' and 'yyy' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "GeneralizedReasoning",
        #   "description": "Schema for evaluating chatbot memory recall by testing its ability to generalize the user's previous reason for change to a new scenario.",
        #   "type": "object",
        #   "properties": {
        #     "User Question": {
        #       "type": "string",
        #       "description": "A new user question where the user inquires about a different activity but with a similar underlying reason as before."
        #     },
        #     "Model Response": {
        #       "type": "string",
        #       "description": "A response demonstrating that the chatbot remembers the previous reason and proactively assumes user preference based on it."
        #     }
        #   },
        #   "required": ["User Question", "Model Response"],
        #   "additionalProperties": false
        # }
    elif action == 'propose_incorrect_reasons_generalization':
        prompt = "Here is the model's response to the user after they mentioned a new activity, where the model accurately connects the user's previous reason for change to this new experience." \
                 "The user's utterance is: " + data['user_utterance'] + "\n\nPrevious reason of change on another activity: " + data['reason_of_change'] + "\n\nThe correct model response: " + data['model_response'] + "\n\n" \
                 "Propose three incorrect responses on purpose to prepare a multiple choice Q&A. Each incorrect option should be a generally good response, but either mentions a wrong reason or completely does not mention the previous reason at all. " \
                 "Each option should share similar tone, matching length, and equal level of detail. Please do NOT be lazy! Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization." \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Fill in the actual data at placeholders "xxx", "yyy", and "zzz" in the template. Please use double quotes for each string. No other words.'
        # schema = {
        #   "title": "IncorrectReasonGeneralizationProposals",
        #   "description": "A list of three incorrect responses that either mention the wrong reason or fail to connect to the user's previously stated reason.",
        #   "type": "array",
        #   "items": {
        #     "type": "string",
        #     "description": "An incorrect response that either connects the wrong reason or completely omits the previous reason."
        #   },
        #   "minItems": 3,
        #   "maxItems": 3
        # }
    elif action == 'ask_previous_reason_after_new_updates':
        prompt = "The user has mentioned the detailed reason below of their preference update in previous conversations:\n\n" + data['event'] + "\n\n" \
                 "You should focus on the [Reasons of Change] part. We actually want to evaluate if the model can remember and utilize this reason of change in the following conversation. " \
                 "Think about the next time the user changes the attitude again, what would the model response? " \
                 "Propose a response that specifically has sensitivity to shifts, and mention how the user still thinks about the previous reason of the previous attitude change. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "Model Response": xxx\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Fill in the actual data at placeholder 'xxx' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "RecallPreviousReason",
        #   "description": "Schema for chatbot response demonstrating awareness of past reasons when the user updates their preferences again.",
        #   "type": "object",
        #   "properties": {
        #     "Model Response": {
        #       "type": "string",
        #       "description": "A chatbot response that recalls the past reason for the user’s previous preference changes while addressing their new update."
        #     }
        #   },
        #   "required": ["Model Response"],
        #   "additionalProperties": false
        # }
    elif action == 'propose_incorrect_reasons_after_new_updates':
        prompt = "Based on this model's response that recalls the correct reason of the user's previous preference changes when the same user changes their preference once again: " + data['response'] + "\n\n" \
                 "Propose three incorrect responses on purpose to prepare a multiple choice Q&A. Each incorrect option should be a generally good response, " \
                 "but either mentions a wrong reason or completely does not mention the previous reason at all. Each option should share similar tone, matching length, and equal level of detail." \
                 "Please do NOT be lazy! Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization. " \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Fill in the actual data at placeholders "xxx", "yyy", and "zzz" in the template. Please use double quotes for each string. No other words.'
        # schema = {
        #   "title": "IncorrectReasonsAfterNewUpdates",
        #   "description": "A list of three incorrect responses that either reference the wrong reason or ignore past reasons altogether.",
        #   "type": "array",
        #   "items": {
        #     "type": "string",
        #     "description": "An incorrect response that either misremembers the past reason or fails to acknowledge any prior preference changes."
        #   },
        #   "minItems": 3,
        #   "maxItems": 3
        # }

    elif action == 'recall_sequence':
        prompt = "We are designing a memory benchmark focused on personalization. Consider the following sequence of user preference changes:\n\n" + data['full_sequence'] + "\n\n" \
                 "The right most one is the most recent update, which the user mentioned that:" + data['user_utterance'] + "\n\n" \
                 "When the user mentions their most recent preference, how should the model respond to demonstrate that it remembers the entire sequence of preference changes, not just the latest one? " \
                 "Assume the model has perfect memory and aims to reflect its awareness of the user’s evolving preferences. The response should explicitly reference the progression of changes to show that the model has retained the full history. " \
                 "Emphasis should be on the sequence of changes rather than the final state of preferences." \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "Model Response": xxx\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'xxx' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "RecallPreferenceSequence",
        #   "description": "Schema for chatbot response demonstrating awareness of the full sequence of user preference changes.",
        #   "type": "object",
        #   "properties": {
        #     "Model Response": {
        #       "type": "string",
        #       "description": "A chatbot response that explicitly references the full sequence of the user’s preference changes rather than just the latest update."
        #     }
        #   },
        #   "required": ["Model Response"],
        #   "additionalProperties": false
        # }
    elif action == 'propose_incorrect_sequence':
        prompt = "Given following the model's response that correctly references the full sequence of preference updates of the user:\n\n" + data['model_response'] + "\n\n" \
                 "Propose three incorrect responses on purpose to prepare a multiple choice Q&A. Each response should look similar, except that they color different incorrect sequence of preference updates. " \
                 "If there is any updates in the sequence, incorrect ones could include incorrect updates or mentions that it is the first time the user mentioned this thing or activity. " \
                 "Do NOT modify the most recent one (the right most one in the sequence). If the sequence has no preference updates, incorrect ones could flip the preference or add one additional change. " \
                 'Each option should share similar tone, matching length, and equal level of detail. ' \
                 'Please do NOT be lazy! Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization.' \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Fill in the actual data at placeholders "xxx", "yyy", and "zzz" in the template. Please use double quotes for each string. No other words.'
        # schema = {
        #   "title": "IncorrectPreferenceSequence",
        #   "description": "A list of three incorrect responses that either misinterpret the preference update sequence or introduce inaccuracies.",
        #   "type": "array",
        #   "items": {
        #     "type": "string",
        #     "description": "An incorrect response that references the preference history inaccurately or contradicts previous changes, ensuring each option is well-written and equal in length to the correct response.",
        #   },
        #   "minItems": 3,
        #   "maxItems": 3
        # }
    elif action == 'extract_object':
        prompt = "You have two tasks. First, please extract the primary noun from the following phrase, ignoring all adjectives or descriptors. Output a single word or short phrase only into the key 'parent_object':\n\n" + data + "\n\n" \
                 "Second, based on the extracted primary noun, propose one different child object name under this parent category, adding some different adjectives or descriptors. Output it into the key 'random_child_object'." \
                 "You should output a dictionary following this format:\n" \
                 "{\n" \
                 '    "parent_object": xxx,\n' \
                 '    "random_child_object": yyy\n' \
                 "}\n" \
                 "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'xxx' and 'yyy' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "ExtractedObject",
        #   "description": "Schema for extracting a parent object and generating a related child object.",
        #   "type": "object",
        #   "properties": {
        #     "parent_object": {
        #       "type": "string",
        #       "description": "The primary noun extracted from the phrase, ignoring adjectives or descriptors."
        #     },
        #     "random_child_object": {
        #       "type": "string",
        #       "description": "A different object name under the same parent category with additional adjectives or descriptors."
        #     }
        #   },
        #   "required": ["parent_object", "random_child_object"],
        #   "additionalProperties": false
        # }

    elif action == 'extract_identity':
        prompt = "Please extract the gender and racial identities from the following persona information. Output a single string. No other words. Here is the full persona:\n\n" + data
        # schema = None
    elif action == 'recommendation':
        prompt = "We aim to assess whether a chatbot can recall a user's most recent preference for a specific type of " + data['parent_object'] + " and provide a personalized recommendation based on this preference. " \
                 "Consider the user's latest preference: " + data['preference'] + " and what they have said: " + data['user_utterance'] + "\n\n" \
                 "Formulate a question the user might ask the chatbot for a recommendation in the future WITHOUT explicitly referencing their previous preferences. " \
                 "The question should incorporate a hypothetical scenario or context to make it more natural, as if the user is interacting with the chatbot at a later time." \
                 "Remember that the user is asking this question to an LLM, not a real human. " \
                 "Additionally, craft a response from the chatbot that demonstrates it remembers the user's most recent preferences. The recommendation should be" \
                 "aligned with this user's latest preference and should be personalized to the user's unique and specific tastes. " \
                 "Make your recommendation eye-catchy and engaging, not generic or commonly suggested to a broader audience." \
                 "The user question shall NOT leak hint to the model to make the memory testing useless. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Question": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Fill in the actual data at placeholders 'xxx' and 'yyy' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
        # schema = {
        #   "title": "UserPreferenceRecommendation",
        #   "description": "Schema for evaluating chatbot memory recall by testing its ability to recommend personalized items based on the user's most recent preferences.",
        #   "type": "object",
        #   "properties": {
        #     "User Question": {
        #       "type": "string",
        #       "description": "A detailed question from the user asking for a recommendation without explicitly referencing past preferences."
        #     },
        #     "Model Response": {
        #       "type": "string",
        #       "description": "A chatbot response that provides a personalized recommendation based on the user's most recent preferences, ensuring it is engaging and unique."
        #     }
        #   },
        #   "required": ["User Question", "Model Response"],
        #   "additionalProperties": false
        # }
    elif action == 'propose_incorrect_recommendations':
        prompt = "Given the following response: " + data['model_response'] + "\n\n to the question: " + data['question'] + "\n\n" \
                 "propose two incorrect responses on purpose to prepare a multiple choice Q&A. "\
                 "Make sure that the incorrect answers are still good suggestions to other users, but just not for this specific user or VIOLATE this user's most recent preferences: " + data['preference'] + \
                 "If the user's preference is about liking something, the incorrect answer should talk about somehow opposite things, as if the model does not remember what this user's preferences are. " \
                 "If the user's preference is about disliking something, the incorrect answer should talk about things this user dislikes. " \
                 'Each option should share similar tone, matching length, and equal level of detail.' \
                 'Please do NOT be lazy! Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization.' \
                 'Output a Python list of two strings, following this format: ["xxx", "yyy"]. Please use double quotes for each sentence. Fill in the actual data at placeholders "xxx" and "yyy" in the template. Do NOT use JSON. No other words.'
        # schema = {
        #     "$schema": "https://json-schema.org/draft/2020-12/schema",
        #     "title": "IncorrectUserRecommendations",
        #     "description": "A list of two incorrect recommendations that either contradict the user's known preferences or suggest something opposite to what they like/dislike.",
        #     "type": "array",
        #     "items": {
        #         "type": "string",
        #         "description": "An incorrect recommendation that does not align with the user's most recent preferences but still sounds like a valid suggestion, ensuring each option is well-written and equal in length to the correct response.",
        #     },
        #     "minItems": 2,
        #     "maxItems": 2
        # }
    elif action == 'propose_stereotypical_recommendation':
        prompt = "Given the following question: " + data['question'] + " and correct response " + data['model_response'] + "\n\n, " \
                 "prepare one incorrect answer that is stereotypical to this user's gender and racial identities, but irrelevant to the specific context " \
                 "and irrelevant to or violate this user's actual preference. Here is the user's identities:\n\n" + data['persona'] + "\n\n"\
                 "Please do NOT be lazy! Make sure this incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization. Output the answer part only using a simple string, like 'xxx'. No additional words. " \
                 "Remember we are creating misleading options in a multiple choice question, so make it sounds like a correct one but do NOT mention that this is actually stereotypical. No other words."
        # schema = None

    elif action == "recall_preference":  # data['Reference'] refers to the two consecutive events
        prompt = "We want to evaluate whether a chatbot can remember user's preferences shared by the user during previous conversations, " \
                 "and whether the model can utilize its memory to provide a personalized response. Given this specific activity\n\n'" + data['Event'] + "\n\n and the user's preferences:\n\n" + data['Preference'] + "\n\nmentioned by the user\n\n" + data['User_Utterance'] + \
                 "we want the user to briefly mention the same activity again in a later conversation, NOT necessary in a question but in a natural sentence, WITHOUT including any preferences. " \
                 "Please then propose a natural model response that specifically acknowledge the user preference and then add something neutral to finish this utterance. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Mention": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify these names of the keys. Fill in the actual data at placeholders 'xxx' and 'yyy' in the template. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == "propose_incorrect_preferences":
        prompt = "This is the correct personalized response to the user utterance '" + data['User_Mention'] + "': " + data['Response'] + "\n\n" \
                 "Please propose three incorrect options. Each incorrect response must cover exactly the same event\n\n" + data['Event'] + "\n\nbut VIOLATE the user’s stated preference in one or more of the following ways: " \
                 "1. Incorrect Acknowledgment of User Preference - acknowledge the opposite preference and then add something neutral to finish this utterance" \
                 "2. Complete Forgetfulness- act as though the user never mentioned their preference and then add something consistent with this ignorance to finish this utterance."
        if data['Old_Response']:
            prompt += "3. Repetition of Old Response - " + data['Old_Response']
        else:
            prompt += "3. Correct Acknowledgment of User Preference but on Incorrect Event - acknowledge the user's preference but on a different event and then add something neutral to finish this utterance"
        prompt += "Each option should share similar tone, matching length, and equal level of detail. Please do NOT be lazy! " \
                  "Make sure each incorrect answer has the same length with the correct one, so that the model can not simply pick the longest answer as the correct one without actual memorization." \
                  'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Fill in the actual data at placeholders "xxx", "yyy", and "zzz" in the template. Please use double quotes for each string. No other words.'

    else:
        raise ValueError("Invalid action", action)
    return prompt


def prompts_for_rewriting_creative_writing(data, persona):
    prompt = "Here is a creative writing sample:\n\n" + data + "\n\nand a new user persona:\n\n" + persona['persona'] + "\n\nand preferences in creative writing:\n\n" + persona['preferences'] + "\n\n" \
             "At this moment, please intentionally rewrite the sample with poor writing and formatting, INCLUDE what are inside this user's [Writing Styles] Dislikes and [Formatting Styles] Dislikes, but REMOVE what are inside this user's [Writing Styles] Likes and [Formatting Styles] Likes, " \
             "the opposite to this user's preference. Just give me the new creative writing sample as a simple string. No other words."
    return prompt


def prompts_for_translating_code(data, persona):
    prompt = "Here is a piece of code in Java:\n\n" + data + "\n\nand a new user persona: " + persona + "\n\nPlease translate this code into Python. " \
             "At this moment, please intentionally write the code with poor coding practices, INCLUDE what are inside this user's [Coding Styles] Dislikes and [Formatting Styles] Dislikes, but REMOVE what are inside this user's [Coding Styles] Likes and [Formatting Styles] Likes, " \
             "the opposite to this user's preference. Just give me the new code formatted with triple backticks and python. No other words."
    return prompt


def prompts_for_rewriting_email(data, persona):
    prompt = "Here is an email sample:\n\n" + data + "\n\nand a new user persona:\n\n" + persona['persona'] + "\n\nand preferences in email writing:\n\n" + persona['preferences'] + "\n\n" \
             "The original email may contain personal information of others, please rewrite them to fit this new user. " \
             "However, at this moment, please intentionally write the email with poor writing and formatting, INCLUDE what are inside this user's [Writing Styles] Dislikes and [Formatting Styles] Dislikes, but REMOVE what are inside this user's [Writing Styles] Likes and [Formatting Styles] Likes, " \
             "the opposite to this user's preference. Just give me the new email as a simple string. No other words."
    return prompt


def prompt_for_preparing_new_content(data, action, data_type):
    if action == 'preferences':
        if data_type == 'coding':
            data_type = 'code implementations in Python'
            prompt = "Here is a new programmer's persona:\n\n" + data + \
                     "\n\nGiven the persona above, please list 5 Python coding styles (e.g., naming conventions, commenting practices, function length, modularity, performance optimization, error handling, readability) and " \
                     "5 Python formatting styles (e.g., indentation, line length, spacing, docstrings, imports organization, inline comments, variable alignment) " \
                     "this programmer may like and dislike, respectively, related to the topic of " + data_type + ". " \
                     "Use bullet points and be specific with short examples. Likes and dislikes should always be clear, objective, and easily verifiable, and all directly related to " + data_type + ". " \
                     "You should output a Python dictionary of the following format:\n\n" \
                     "{\n" \
                     '   "[Coding Styles] Likes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                     '   "[Coding Styles] Dislikes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                     '   "[Formatting Styles] Likes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                     '   "[Formatting Styles] Dislikes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx}\n' \
                     "}\n\n" \
                     "Do NOT modify the names of these keys. Please use double quotes for each key and value. No other words."
        elif data_type == 'writing' or data_type == 'email':
            prompt = "Here is a new author's persona:\n\n" + data + "\n\nGiven the persona above, please list 5 writing styles (e.g., tone, wording, emojis, valence, arousal, dominance, personality, and etc) and " \
                     "5 formatting styles (e.g., subsections, signature, final closing, title, side notes, paragraph length, and ways to write first & last names, abbreviation, time, and etc) " \
                     "this writer may like and dislike, respectively, related to the topic of " + data_type + ", " \
                     "using bullet points. Always be specific using short examples. Likes and dislikes should always be clear, objective, and easily verifiable, and all directly related to " + data_type + \
                     "You should output a Python dictionary of the following format:\n\n" \
                     "{\n" \
                     '   "[Writing Styles] Likes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                     '   "[Writing Styles] Dislikes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                     '   "[Formatting Styles] Likes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                     '   "[Formatting Styles] Dislikes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                    "}\n" \
                    "Do NOT modify the names of these keys.  Please use double quotes for each key and value. No other words."
        else:
            raise ValueError("Invalid data type", data_type)
    elif action == 'rewrite_from_persona':
        prompt = "Here is a " + data_type + " sample:\n\n" + data + "\n\nGiven the " + data_type + " sample and the persona above. "
        if data_type == 'coding':
            prompt += "Please modify some sentences and formats as if it was written by the author with this new persona, correctly reflecting EVERY SINGLE like and avoiding EVERY SINGLE dislike in coding and formatting styles. " \
                      "You can add comments explaining the modifications using the first-person perspective. " \
                      "You should only output the rewritten code, formatted with triple backticks and python. No other words."
        elif data_type == 'writing' or data_type == 'email':
            prompt += "Please modify some sentences and formats as if it was written by the author with this new persona, correctly reflecting EVERY SINGLE like and avoiding EVERY SINGLE dislike in writing and formatting styles. " \
                      "Within the new sample, before each sentence you wanna modify, make sure to add a '[Side_Note]' in square brackets explaining why this modification is aligned with what " + data_type + " or formatting persona points of this new author. " \
                      "Do NOT make any modifications on other sentences whose " + data_type + " or formatting styles are not related to the new author's persona, keeping them word-by-word identical. " \
                      "You should only output the rewritten sample as a simple string. No other words."
    elif action == 'rewrite_as_conversation':
        if data_type == 'coding':
            prompt = "Given the original and rewritten programming code above, create a conversation record as if the programmer is consulting an AI coding assistant to help refactor and improve the original code into the rewritten version. " \
                     "The programmer should explicitly express what they like and dislike about coding style and formatting styles. " \
                     "Each concern should be linked to '[Coding Styles] Likes', '[Coding Styles] Dislikes', '[Formatting Styles] Likes', '[Formatting Styles] Dislikes' listed in the persona above. " \
                     "The assistant should recommend improvements, leading to the final rewritten version. However, if the assistant suggests a different approach first, the programmer should express their dissatisfaction and explain why, leading to the final version. " \
                     "The user and assistant should explicitly display modified code snippets in their discussion. The whole conversation should be long enough to cover all changes in the rewritten sample. " \
                     "The user should also say their preferences mentioned in the Side_Note, as if the AI assistant can not see those Side_Note. " \
                     "Except for the very first two sentences where the user explains how they want the assistant to help them with the programming code, follow this format:\n\n" \
                     "[\n" \
                     '"[Original_Code]: xxx", (Code should be formatted with triple backticks and python. The mark [Original_Code] should appear once at each code section the user mentions, not every line)\n' \
                     '"[Side_Note]: [Coding Styles] Likes OR [Coding Styles] Dislikes OR [Formatting Styles] Likes OR [Formatting Styles] Dislikes xxx", (details here)\n' \
                     '"User: xxx",\n (User utterance should be text, WITHOUT triple backticks)' \
                     '"Assistant: xxx",\n (New code should be formatted with triple backticks and python. )' \
                     '"User: xxx",\n' \
                     "...,]\n" \
                     "Output the full conversation as a python list of strings, where each element in the list is one utterance or one piece of the original code. Use double quotes for each string. Do NOT change the names before the colon mark. No other words."
        elif data_type == 'writing' or data_type == 'email':
            prompt = "Given the original and rewritten writing samples above, create a conversation record as if the new author is consulting an expert AI writing assistant to help the author convert the original sample to the rewritten sample. " \
                     "The author should propose questions and concerns, explicitly saying that they likes and dislikes regarding the writing and formatting styles. We need to see every explicit and concrete reasons, " \
                     "and you should always use a '[Side_Note]' with square brackets to link each modification to its corresponding '[Writing Styles] Likes', '[Writing Styles] Dislikes', '[Formatting Styles] Likes', and '[Formatting Styles] Dislikes' listed in the persona above. " \
                     "The assistant should give recommendations that result in the modified sentences in the rewritten sample, but it could also propose a different suggestion, the author dislikes it and says why, and the assistant finally propose the one shown in the final rewritten sample." \
                     "Make sure to explicitly include each pair of original and modified sentence in the conversation, as if these two persons are showing the sentence to each other in a conversation. " \
                     "The whole conversation should be long enough to cover all modified sentences in the rewritten sample." \
                     "The user should also say their preferences mentioned in the Side_Note, as if the AI assistant can not see those Side_Note. " \
                     "Except for the very first two sentences where the user explains how they want the assistant to help them with the writing, you should follow this format for the conversation:\n\n" \
                     "[\n" \
                     '"[Original_Sentence]: xxx",\n' \
                     '"[Side_Note]: [Writing Styles] Likes OR [Writing Styles] Dislikes OR [Formatting Styles] Likes OR [Formatting Styles] Dislikes xxx", (details here) \n' \
                     '"User: xxx",\n' \
                     '"Assistant: xxx",\n' \
                     '"User: xxx",\n' \
                     "...,]\n" \
                     "Output the full conversation as a python list of strings, where each line is string. Use double quotes for each string. Do NOT change the names before the colon mark. No other words."
        else:
            raise ValueError("Invalid data type", data_type)
    else:
        raise ValueError("Invalid action", action)
    return prompt


def prompt_for_content_generation(data, action):
    if action == 'write_new_sample':
        prompt = "The writer's conversation record with the writing assistant:\n\n" + data + "\n\n" \
                 "Given the conversation above, your task is to write a new creative writing paragraph of at most 5 sentences that directly and explicitly aligns with the personas, likes, and dislikes in writing and formatting styles." \
                 "You should simply output the new paragraph as a string. No other words."
    elif action == 'write_violating_sample':
        prompt = "The writer's conversation record with the writing assistant:\n\n" + data + "\n\n" \
                 "You have written a paragraph that aligns with the personas. Next, given the same conversation, please write a new creative writing paragraph of at most 5 sentences that, on-purposely, violates the personas, likes, and dislikes in writing and formatting styles." \
                 "You should simply output the new paragraph as a string. No other words."
    elif action == 'write_new_sample_oracle':
        if data['topic'] == 'writing':
            prompt = "The writer's persona:\n\n" + data['persona'] + "\n\nand the likes and dislikes in writing and formatting styles:\n\n" + data['preferences'] + "\n\n" \
                     "Given the information above about the writer, your task is to write a new creative writing paragraph of at least 10 sentences that directly and explicitly aligns with the personas, likes, and dislikes in writing and formatting styles." \
                     "You should simply output the new paragraph as a string. No other words."
        elif data['topic'] == 'coding':
            prompt = "The programmer's persona:\n\n" + data['persona'] + "\n\nand the likes and dislikes in coding and formatting styles:\n\n" + data['preferences'] + "\n\n" \
                     "Given the information above about the programmer, your task is to write a new piece of Python code that directly and explicitly aligns with the personas, likes, and dislikes in coding and formatting styles." \
                     "You should simply output the new code formatted with triple backticks and python. No other words."
        elif data['topic'] == 'email':
            prompt = "The email writer's persona:\n\n" + data['persona'] + "\n\nand the likes and dislikes in writing and formatting styles:\n\n" + data['preferences'] + "\n\n" \
                     "Given the information above about the email writer, your task is to write a new email of at least 10 sentences that directly and explicitly aligns with the personas, likes, and dislikes in writing and formatting styles." \
                     "You should simply output the new email as a string. No other words."
        else:
            raise ValueError("Invalid topic", data['topic'])
    else:
        raise ValueError("Invalid action", action)
    return prompt


def prompt_for_evaluating_content(data, action):
    if action == 'evaluate_aligned':
        prompt = "Here is the writer's persona:\n\n" + data['persona'] + "\n\nThe writer's likes and dislikes on writing and formatting styles:\n\n" + data['preferences'] + "\n\n" \
                 "Paragraph 1:\n\n" + data['paragraph1'] + "\n\nParagraph 2:\n\n" + data['paragraph2'] + "\n\n" \
                 "Your tasks are to find how many sentences in Paragraph 1 and Paragraph 2 respectively that align with the authors' persona, likes, and dislikes. " \
                 'Only mention those that are aligned, using a JSON file with two keys "Paragraph_1" and "Paragraph_2", whose value is a list of Python dictionary.' \
                 'For each sentence included in the list, add the key "Reason" BEFORE the "Sentence"m explaining why, i.e., what persona, likes, and dislikes it is aligned with.' \
                 "Here is the template your output should follow:\n\n" \
                 "{\n" \
                 '  "Paragraph_1": "[\n' \
                 "      {\n" \
                 '          "Reason": "xxx",\n' \
                 '          "Sentence": "yyy"\n' \
                 "      },\n...\n" \
                 "  ],\n" \
                 '  "Paragraph_2": "[\n' \
                 "      {\n" \
                 '          "Reason": "xxx",\n' \
                 '          "Sentence": "yyy"\n' \
                 "      },\n...\n" \
                 "  ],\n" \
                 "}" \
                 "Do NOT modify the names of these keys. No other words."
    elif action == 'evaluate_violated':
        prompt = "Same as above, but list sentences that VIOLATE this writer's persona, likes, and dislikes, if any. You should follow the same template below:\n\n" \
                 "{\n" \
                 '  "Paragraph_1": "[\n' \
                 "      {\n" \
                 '          "Reason": "xxx",\n' \
                 '          "Sentence": "yyy"\n' \
                 "      },\n...\n" \
                 "  ],\n" \
                 '  "Paragraph_2": "[\n' \
                 "      {\n" \
                 '          "Reason": "xxx",\n' \
                 '          "Sentence": "yyy"\n' \
                 "      },\n...\n" \
                 "  ],\n" \
                 "}" \
                 "Do NOT modify the names of these keys. No other words."
    else:
        raise ValueError("Invalid action", action)
    return prompt