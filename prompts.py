import torch
import random


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
             "All events must have an appropriate time stamp in the format of MM/DD/YYYY. List at least 10 events, more are welcome. " \
             "Here is the template you should follow for each event:\n\n" \
                '"MM/DD/YYYY": {\n' \
                    '"Event": xxx, \n' \
                    '"Category": "Short-Term" OR "Long-Term"\n' \
                    '"[Fact] Likes" OR "[Fact] Dislikes": xxx, \n' \
                "}, \n\n" \
             "Do NOT modify the names of these keys. Please use double quotes for the names." \
             "Here is the persona: " + persona
    return prompt


def prompts_for_init_contextual_personal_history(context, start_time, persona, general_personal_history):
    prompt = "Here is the persona:\n\n" + persona + "\n\nHere are 10 events related to the person's general background history:\n\n" + general_personal_history + "\n\n" \
             "Given the persona and the person's general background history above, continue to list 10 personal hobbies and 10 things this person dislikes but others might like, using bullet points, related to " + context + ". " \
             "Next, write 10 more events related to the context of " + context + ". Include all these 20 new things this person likes and dislikes, and rewrite them as appropriate events." \
             "Do NOT mention anything already mentioned above. Do NOT mention anything about the general personal history, like the professional development. " \
             "Each event must come with the related personal hobbies or dislikes, marked using a key '[Fact] Likes:' or '[Fact] Dislikes:'." \
             "Use the same JSON format with MM/DD/YYYY timestamp from " + start_time + ", and use short-term/long-term labels as above. There should be 10 short-term and 10 long-term events." \
             "Here is the template you should follow for each event:\n\n" \
                '"MM/DD/YYYY": {\n' \
                    '"Event": xxx, \n' \
                    '"Category": "Short-Term" OR "Long-Term"\n' \
                    '"[Fact] Likes" OR "[Fact] Dislikes": xxx, \n' \
                "}, \n\n" \
             "Do NOT modify the names of these keys. Please use double quotes for the names."
    return prompt


def prompts_for_expanding_personal_history(context=None, type='general', period='WEEK'):
    if type != 'general':
        assert context is not None

    if type == 'general':
        prompt = "Given the initial general personal history, think about what would happen to the same person in a " + period + ". "
    else:
        prompt = "Given the initial contextual personal history, think about what would happen to the same person in a " + period + " related to the " + context + ". "
    prompt += "More than half of those new points could be, though logically still make sense, but contradictory to the original persona and personal history, especially those ['Short-Term'] facts." \
              "If there is any contradictions or knowledge updates, remember to include why, i.e., the user's reasons and intentions using an additional key '[Reasons of Change]'. " \
              "Try finding unique reasons for this person, not common for the general public, that trigger the change. " \
              "Please also use the following keys, and do NOT modify the name of these keys:\n\n" \
              "The key '[Old Event]' to mention the related old event contradictory to it, the key '[Old Event Date]' to mention its timestamp MM/DD/YYYY, " \
              "the key '[Old Fact] Likes' or '[Old Fact] Dislikes' to mention the underlying like or dislike of this peron." \
              "the key '[Updated Fact] Likes' or '[Updated Fact] Dislikes' should be exactly the opposite of the '[Old Fact] Likes' or '[Old Fact] Dislikes'." \
              "If this is a new event without contradiction with previous ones, marked related personal hobbies or dislikes using the key '[Fact] Likes:' or '[Fact] Dislikes:', but do NOT include the '[Reasons of Change]' key.\n\n" \
              "Any contradictions should focus on what this person prefers and dislikes. " \
              "You shall also include some contradictions to the existing contradictions in the previous history, back and forth. For example, the person may like one thing, dislike it, and in some cases come back to like it again." \
              "Now, please continue to write 10 more events aligned with this persona. Do NOT repeat anything already mentioned above. " \
              "Use the same JSON format with MM/DD/YYYY timestamp starting at the end of the previous general personal history, and use short-term/long-term labels as above. There should be 5 short-term and 5 long-term events."

    prompt += "Here is the template you should follow for each event WITHOUT knowledge updates:\n\n" \
              '"MM/DD/YYYY": {\n' \
                  '"Event": xxx, \n' \
                  '"Category": "Short-Term" OR "Long-Term"\n' \
                  '"[Fact] Likes" OR "[Fact] Dislikes": xxx, \n' \
              "}, \n\n" \
              "Here is the template you should follow for each event WITH knowledge updates:\n\n" \
              "'MM/DD/YYYY': {\n" \
                  '"Event": xxx, \n' \
                  '"Category": "Short-Term" OR "Long-Term"\n' \
                  '"[Reasons of Change]": xxx, \n' \
                  '"[Updated Fact] Likes" OR "[Updated Fact] Dislikes": xxx, \n' \
                  '"[Old Fact] Likes" OR "[Old Fact] Dislikes": xxx, \n' \
                  '"[Old Event Date]": MM/DD/YYYY, \n' \
                  '"[Old Event]": xxx, \n' \
              "}\n" \
              "Do NOT modify the names of these keys. Please use double quotes for the names."
    return prompt


def prompts_for_generating_conversations(context, persona, curr_personal_history=None, period='INIT'):
    if context == 'therapy':
        context_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif context == 'legal':
        context_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    else:
        context_name, user, agent = context, 'User', 'Assistant'

    prompt = "Your task is to rewrite the following list of events related to a personal history as a format of conversation record under the context of " + context_name + ". " \
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
              "...] Use a Python list of strings where each sentence is one string. Use double quotes for each sentence. Do NOT use JSON. No other words."
    return prompt


def prompts_for_reflecting_conversations(context, data, round, period='INIT'):
    if context == 'therapy':
        context_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif context == 'legal':
        context_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    else:
        context_name, user, agent = context, 'User', 'Assistant'

    if period == 'INIT':
        history_block = "'Init General Personal History'"
        conversation_block = "'Init Conversation'"
    elif period == 'WEEK':
        history_block = "'General Personal History Next Week'"
        conversation_block = "'Conversation Next Week'"
    elif period == 'MONTH':
        history_block = "'General Personal History Next Month'"
        conversation_block = "'Conversation Next Month'"
    else:
        history_block = "'General Personal History Next Year'"
        conversation_block = "'Conversation Next Year'"

    if round == 1:
        prompt = "Given the following " + history_block + " and the " + conversation_block + ", check if the " + conversation_block + " has covered every single timestamp in the " + history_block + ". " \
                 "List all missed ones:\n\n" + data['history_block'] + "\n\n" + data['conversation_block']
    elif round == 2:
        prompt = "Please fill in these missed timestamps with their corresponding events mentioned in the " + history_block + " into the " + conversation_block + ". " \
                 "You may add some transition sentences to make it smooth, but do NOT modify any other words in the original conversation. Keep them word-by-word IDENTICAL." \
                 "If there is no missed timestamp, no need to change any part of the original conversation. Follow exactly the SAME template in the original conversation:\n\n" \
                 "[\n" \
                 '"Side_Note: [xxx] MM/DD/YYYY",' \
                 '"' + user + ': yyy",' \
                 '"' + agent + ': zzz",' \
                 "...] Use a Python list of strings where each sentence is one string. Use double quotes for each sentence. Do NOT use JSON. Just output the completed conversation. No other words."
    else:
        raise ValueError("Invalid round", round)
    return prompt


def prompts_for_expanding_conversation_section(context, section):
    if context == 'therapy':
        context_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif context == 'legal':
        context_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    else:
        context_name, user, agent = context, 'User', 'Assistant'

    prompt = "Please expand these sentences. I do NOT want any new user preferences, examples, or changes to the story behind the conversation. " \
             "Instead, extend each line to AT LEAST FIVE sentences by adding additional details or irrelevant context that delves deeper into the mentioned objects or events. " \
             "Ensure that no new preferences are introduced or altered. Each revised sentence should provide greater depth while maintaining consistency with the original narrative and intent." \
             "Note that the lines said by " + agent + " should be even longer to show the caring or professionalism. " \
             "Also note that if the last line is another line of 'Side_Note', that 'Side_Note' indicates the next event, so the previous line should consider how to smoothly transit the conversation. " \
             "Here is the section you should expand, while do NOT expand or modify the line(s) of Side_Note.\n\n" + '\n'.join(section) + "\n\n" \
             "Follow exactly the SAME template in the original sentences:\n\n" \
             "[\n" \
             '"Side_Note: [xxx] MM/DD/YYYY",' \
             '"' + user + ': yyy",' \
             '"' + agent + ': zzz",' \
             "...] Use a Python list of strings where each sentence is one string. Use double quotes for each sentence. Do NOT use JSON. Just output the expanded conversation. No other words."
    return prompt


def prompts_for_generating_qa(data, action):
    if action == 'factual_qa':
        prompt = "Please propose one factual Q&A for this event happened to this specific " + data['user'] + ", in order to evaluate the model's memory capabilities. " \
                 "Please write the new question-answer pair in JSON format, with keys 'Question' and 'Answer'. " \
                 "The question should explicitly include the timestamp " + data['timestamp'] + ". Follow this format:\n" \
                 "{\n" \
                 '    "Question": xxx,\n' \
                 '    "Answer": yyy\n' \
                 "}" \
                 "Do NOT modify the names of these keys. Please use double quotes for the names. No other words." \
                 "Here is the event:\n\n" + data['event']
    elif action == 'propose_incorrect_facts':
        prompt = 'Given the following Q&A, prepare three incorrect answers.\n\n' + data + 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. ' \
                 "Incorrect answers should have the same length with the correct answer."
    elif action == 'abstention':
        prompt = "Given this question '" + data + "', your next task is to rewrite the object or event name mentioned in this question to a similar, but irrelevant name. " \
                 "The purpose of this step is to evaluate if the model can correctly remember that the new name has actually never been mentioned. Do NOT modify other parts of the question. " \
                 "Output both the new question and the new object or event name, following this format:\n" \
                 "{\n" \
                 '    "New Question": xxx,\n' \
                 '    "New Name": yyy\n' \
                 "}" \
                 "Do NOT modify the names of these keys. Please use double quotes for the names. No other words."
    elif action == 'propose_incorrect_reasons':
        prompt = 'Given the following Q&A, prepare three incorrect answers.\n\n' + data + 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. ' \
                 "Incorrect answers should have the same length with the correct answer."
    elif action == 'extract_object':
        prompt = "You have two tasks. First, please extract the primary noun from the following phrase, ignoring all adjectives or descriptors. Output a single word or short phrase only into the key 'parent_object':\n\n" + data + "\n\n" \
                 "Second, based on the extracted primary noun, propose one different child object name under this parent category, adding some different adjectives or descriptors. Output it into the key 'random_child_object'." \
                 "You should output a dictionary following this format:\n" \
                 "{\n" \
                 '    "parent_object": xxx,\n' \
                 '    "random_child_object": yyy\n' \
                 "}\n" \
                 "Do NOT modify the names of these keys. Please use double quotes for the names. No other words."
    elif action == 'recommendation':
        prompt = "What recommendation about " + data['parent_object'] + " would you give to this specific " + data['user'] + ", but NOT to other common " + data['user'] + " in general? " \
                 "Your recommendation should align with this " + data['user'] + "'s most up-to-date preferences towards " + data['parent_object'] + "." \
                 "Please first construct a hypothetical scenario or context where this " + data['user'] + " would need your recommendation, and then provide a concise and aligned suggestion. " \
                 "Say your recommendations directly without explanations or using words like 'I would recommend'. If the answer is a single phrase, add a little descriptions. " \
                 "Please write the new question-answer pair in JSON format, with keys 'Question' and 'Answer'. " \
                 "{\n" \
                 '    "Question": xxx,\n' \
                 '    "Answer": yyy\n' \
                 "}" \
                 "Do NOT modify the names of these keys. Please use double quotes for the names. No other words." \
                 "Here are this " + data['user'] + "'s most recent events:\n\n" + data['events']
    elif action == 'propose_incorrect_recommendations':
        prompt = 'Given the following Q&A, prepare two incorrect answers for the multiple-choice question.\n\n' + data['qa'] + '\n\nOutput a Python list of two strings, following this format: ["xxx", "yyy"]. Please use double quotes for each sentence. Do NOT use JSON.' \
                 "Make sure that the incorrect answers are still good suggestions to other users, but just not for this specific " + data['user'] + " or violate this " + data['user'] + "'s preferences. " \
                 "Follow the same language and length as the correct answer. These two options should be different. Remember we are creating misleading options, so do NOT mention that this is not aligned with the " + data['user'] + " preferences. No other words."
    elif action == 'extract_identity':
        prompt = "Please extract the gender and racial identities from the following persona information. Output a single string. No other words. Here is the full persona:\n\n" + data
    elif action == 'propose_stereotype_recommendation':
        prompt = "Given the following Q&A, prepare one incorrect answer that is stereotypical to this " + data['user'] + "'s gender and racial identities, but irrelevant to the specific context" \
                 "and irrelevant to or violate this " + data['user'] + "'s actual preference. Here is the question and the correct answer:\n\n" + data['qa'] + "\n\nHere is the " + data['user'] + "'s identities:\n\n" + data['persona'] + "\n\n"\
                 "Follow the same length as the correct answer. Output the answer part only using a simple string, like 'xxx'. No additional words. Remember we are creating misleading options, so do NOT mention that this is not aligned with the " + data['user'] + " preferences. No other words."
    else:
        raise ValueError("Invalid action", action)
    return prompt


def prompt_for_preparing_new_content(data, action):
    if action == 'preferences':
        prompt = "Here is a new author's persona:\n\n" + data + "\n\nGiven the persona above, please list 5 writing styles (e.g., tone, wording, emojis, valence, arousal, dominance, personality, and etc) and " \
                 "5 formatting styles (e.g., subsections, signature, final closing, title, side notes, paragraph length, and ways to write first & last names, abbreviation, time, and etc) this writer may likes and dislikes, respectively, " \
                 "using bullet points. You should output a Python dictionary of the following format:\n\n" \
                 "{\n" \
                 '   "[Writing Styles] Likes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                 '   "[Writing Styles] Dislikes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                 '   "[Formatting Styles] Likes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                 '   "[Formatting Styles] Dislikes": {"1": xxx, "2": xxx, "3": xxx, "4": xxx, "5": xxx},\n' \
                "}\n" \
                "Do NOT modify the names of these keys. No other words."
    elif action == 'rewrite_from_persona':
        prompt = "Here is a creative writing sample:\n\n" + data + "\n\nGiven the creative writing sample and the persona above, " \
                 "please modify some sentences and formats as if it was written by the author with this new persona, incorporating all likes and dislikes in writing and formatting styles. " \
                 "Do NOT make any modifications on other sentences whose writing or formatting styles are not related to the new author's persona, keeping them word-by-word identical." \
                 "Within the new sample, before each sentence you wanna modify, make sure to add a '[Side_Note]' in square brackets explaining why this modification is aligned with what writing or formatting persona points of this new author. " \
                 "You should only output the rewritten sample as a simple string. No other words."
    elif action == 'rewrite_as_conversation':
        prompt = "Given the original and rewritten samples above, create a conversation record as if the new author is consulting an expert writing assistant to help the author convert the original sample to the rewritten sample. " \
                 "The author should propose questions and concerns, explicitly saying that they likes and dislikes regarding the writing and formatting styles. We need to see every explicit and concrete reasons, " \
                 "and you should always use a '[Side_Note]' with square brackets to link each modification to its corresponding '[Writing Styles] Likes', '[Writing Styles] Dislikes', '[Formatting Styles] Likes', and '[Formatting Styles] Dislikes' in the persona. " \
                 "The assistant should give recommendations that result in the modified sentences in the rewritten sample, but it could also propose a different suggestion, the author dislikes it and says why, and the assistant finally propose the one shown in the final rewritten sample." \
                 "Make sure to explicitly include each pair of original and modified sentence in the conversation, as if these two persons are showing the sentence to each other in a conversation. " \
                 "Each utterance in the conversation should be short, like a in-person consultation, but the whole conversation should be long enough to cover all modified sentences in the rewritten sample." \
                 "Except for the very first two sentences where the user explains how they want the assistant to help them, you should follow this format for the conversation:\n\n" \
                 "[Original_Sentence]: xxx\n" \
                 "[Side_Note]: '[Writing Styles] Likes' OR '[Writing Styles] Dislikes' OR '[Formatting Styles] Likes' OR '[Formatting Styles] Dislikes' xxx (details here) \n" \
                 "User: xxx\n" \
                 "Assistant: xxx\n" \
                 "User: xxx\n" \
                 "Output a Python list of strings, where each line is a string. Do NOT change the names before the colon mark. No other words."
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
        prompt = "The writer's persona:\n\n" + data['persona'] + "\n\nand the likes and dislikes in writing and formatting styles:\n\n" + data['preferences'] + "\n\n" \
                 "Given the information above about the writer, your task is to write a new creative writing paragraph of at most 5 sentences that directly and explicitly aligns with the personas, likes, and dislikes in writing and formatting styles." \
                 "You should simply output the new paragraph as a string. No other words."
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