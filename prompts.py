import torch
import random


def prompts_for_background_data(content):
    prompt = "Please look at the following conversation and understand its content, format, length, and style:\n\n" + content
    return prompt


def prompts_for_random_question_follow_up():
    prompt = "Find a follow-up question based on the previous question and response. Output the question only. No other words."
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
             "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format." \
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
             "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format."
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
              "Try finding some very unique and personal reasons for this person, uncommon for the general public, that trigger the change. " \
              "Please also use the following keys, and do NOT modify the name of these keys:\n\n" \
              "The key '[Old Event]' to mention the related old event contradictory to it, the key '[Old Event Date]' to mention its timestamp MM/DD/YYYY, " \
              "the key '[Old Fact] Likes' or '[Old Fact] Dislikes' to mention the underlying like or dislike of this peron." \
              "the key '[Updated Fact] Likes' or '[Updated Fact] Dislikes' should be exactly the OPPOSITE to its corresponding '[Old Fact] Likes' or '[Old Fact] Dislikes'." \
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
                  '"[Reasons of Change]": xxx, (Please find some unique, uncommon, and personal reasons!) \n' \
                  '"[Updated Fact] Likes" OR "[Updated Fact] Dislikes": xxx, \n' \
                  '"[Old Fact] Likes" OR "[Old Fact] Dislikes": xxx, \n' \
                  '"[Old Event Date]": MM/DD/YYYY, \n' \
                  '"[Old Event]": xxx, \n' \
              "}\n" \
              "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format."
    return prompt


def prompts_for_generating_conversations(context, persona, curr_personal_history=None, period='INIT'):
    if topic == 'therapy':
        topic_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif topic == 'legal':
        topic_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    else:
        topic_name, user, agent = context, 'User', 'Assistant'

    prompt = "Your task is to rewrite the following list of events related to a personal history as a format of conversation record under the context of " + topic_name + ". " \
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
    if topic == 'therapy':
        topic_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif topic == 'legal':
        topic_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    else:
        topic_name, user, agent = context, 'User', 'Assistant'

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


def prompts_for_expanding_conversation_section(context, data):
    if topic == 'therapy':
        topic_name, user, agent = 'therapy', 'Patient', 'Therapist'
    elif topic == 'legal':
        topic_name, user, agent = 'legal consulting', 'Client', 'Lawyer Assistant'
    else:
        topic_name, user, agent = context, 'User', 'Assistant'

    prompt = "Please expand these sentences. I do NOT want any new user preferences, examples, or changes to the story behind the conversation. " \
             "Instead, extend each line to AT LEAST FIVE sentences by adding additional details or irrelevant context that delves deeper into the mentioned objects or events. " \
             "Ensure that no new preferences are introduced or altered. Each revised sentence should provide greater depth while maintaining consistency with the original narrative and intent." \
             "Note that the lines said by " + agent + " should be even longer to show the caring or professionalism. " \
             "Also note that if the last line is another line of 'Side_Note', that 'Side_Note' indicates the next event, so the previous line should consider how to smoothly transit the conversation. " \
             "Here is the section you should expand, while do NOT expand or modify the line(s) of Side_Note.\n\n" + '\n'.join(data['section']) + "\n\n" \
             "Please remove or rephrase any timestamp MM/DD/YYYY mentioned by the " + user + " and " + agent + " in their utterances. Note that this conversation is happening at " + data['last_timestamp'] + "." \
             "But you should keep the Side_Note unmodified. Each Side_Note should include the original timestamp MM/DD/YYYY. " \
             "Follow exactly the SAME template in the original sentences:\n\n" \
             "[\n" \
             '"Side_Note: [...] MM/DD/YYYY" (Please include MM/DD/YYYY here),' \
             '"' + user + ': yyy" (Do NOT include MM/DD/YYYY here),' \
             '"' + agent + ': zzz",' \
             "...] Use a Python list of strings where each sentence is one string. Use double quotes for each sentence. Do NOT use JSON. Just output the expanded conversation. No other words."
    return prompt


def prompts_for_generating_qa(data, action):
    if action == 'recall_facts':
        prompt = "We want to evaluate whether a chatbot can remember factual information (NOT the user's preferences toward it) shared by the user during previous conversations, " \
                 "and whether the model can utilize its memory to provide a personalized response. Given this specific activity described by the user in a conversation with the chatbot:\n\n" + data['event'] + "\n\n" \
                 "What question might the user query the chatbot model to bring up this topic again? Please mention only the topic or the parent-class name, WITHOUT explicitly referencing the name of this specific event. " \
                 "Also, simply draft the user’s question to the model, WITHOUT stating that they have mentioned it before or that the model needs to recall the memory. " \
                 "Make the user question more detailed with some context. Remember that the user is asking this question to an LLM, not a real human. " \
                 "Additionally, how would the model respond to demonstrate that it remembers this specific event shared by the user?" \
                 "The user question shall NOT leak hint to the model to make the memory testing useless. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Question": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'propose_incorrect_facts':
        prompt = "This is the correct personalized response to the question: " + data['question'] + ": " + data['response'] + "\n\n" \
                 "Please propose three incorrect options to prepare a multiple choice Q&A, keeping all incorrect responses generally good but mentioning different things or activities. " \
                 "Each option should share similar tone, matching length, and equal level of detail." \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'
    elif action == 'recall_facts_inverse':
        prompt = "We want to evaluate whether a chatbot can remember factual information (NOT the user's preferences toward it) shared by the user during previous conversations, " \
                 "and whether the model can utilize its memory to provide a personalized response. Given this specific activity described by the user in a conversation with the chatbot:\n\n" + data['event'] + "\n\n" \
                 "What question might the user ask the chatbot to bring up this topic again? Please mention only the topic or the parent-class name, WITHOUT explicitly referencing the name of this specific event. " \
                 "Also, simply mimic the user’s question, WITHOUT stating that they have mentioned it before or that the model needs to recall the memory." \
                 "Most importantly, the user should say they want to try something new, WITHOUT explicitly saying what they have done before to test the model's memory capability. " \
                 "Make the user question more detailed with some context. Remember that the user is asking this question to an LLM, not a real human. " \
                 "Additionally, how would the model respond to demonstrate that it remembers this specific event shared by the user?" \
                 "The model's response should simply give an answer, WITHOUT first mentioning what the user has already done before. " \
                 "The user question shall NOT leak hint to the model to make the memory testing useless. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "User Question": xxx,\n' \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'propose_incorrect_facts_inverse':
        prompt = "Given this question from the user: " + data['question'] + ", please create three responses inspired by these conversations from other users. " \
                 "Since they originate from other users, it is safe to use them here.\n\n" + data['random_event_histories'] + "\n\n" \
                 "Each option should share similar tone, matching length, and equal level of detail. " \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'

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
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'propose_incorrect_reasons_generalization':
        prompt = "Here is the model's response to the user after they mentioned a new activity, where the model accurately connects the user's previous reason for change to this new experience." \
                 "The user's utterance is: " + data['user_utterance'] + "\n\nPrevious reason of change on another activity: " + data['reason_of_change'] + "\n\nThe correct model response: " + data['model_response'] + "\n\n" \
                 "Propose three incorrect responses on purpose to prepare a multiple choice Q&A. Each incorrect option should be a generally good response, but either mentions a wrong reason or completely does not mention the previous reason at all. " \
                 "Each option should share similar tone, matching length, and equal level of detail. " \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'
    elif action == 'ask_previous_reason_after_new_updates':
        prompt = "The user has mentioned the detailed reason below of their preference update in previous conversations:\n\n" + data['event'] + "\n\n" \
                 "You should focus on the [Reasons of Change] part. We actually want to evaluate if the model can remember and utilize this reason of change in the following conversation. " \
                 "Think about the next time the user changes the attitude again, what would the model response? " \
                 "Propose a response that specifically has sensitivity to shifts, and mention how the user still thinks about the previous reason of the previous attitude change. " \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'propose_incorrect_reasons_after_new_updates':
        prompt = "Based on this model's response that recalls the correct reason of the user's previous preference changes when the same user changes their preference once again: " + data['response'] + "\n\n" \
                 "Propose three incorrect responses on purpose to prepare a multiple choice Q&A. Each incorrect option should be a generally good response, " \
                 "but either mentions a wrong reason or completely does not mention the previous reason at all. Each option should share similar tone, matching length, and equal level of detail." \
                 'Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'

    elif action == 'recall_sequence':
        prompt = "We are designing a memory benchmark focused on personalization. Consider the following sequence of user preference changes:\n\n" + data['full_sequence'] + "\n\n" \
                 "The right most one is the most recent update, which the user mentioned that:" + data['user_utterance'] + "\n\n" \
                 "When the user mentions their most recent preference, how should the model respond to demonstrate that it remembers the entire sequence of preference changes, not just the latest one? " \
                 "Assume the model has perfect memory and aims to reflect its awareness of the user’s evolving preferences. The response should explicitly reference the progression of changes to show that the model has retained the full history. " \
                 "Emphasis should be on the sequence of changes rather than the final state of preferences." \
                 "Always follow the template below:\n\n" \
                 "{\n" \
                 '    "Model Response": yyy\n' \
                 "}. " \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'propose_incorrect_sequence':
        prompt = "Given following the model's response that correctly references the full sequence of preference updates of the user:\n\n" + data['model_response'] + "\n\n" \
                 "Propose three incorrect responses on purpose to prepare a multiple choice Q&A. Each response should look similar, except that they color different incorrect sequence of preference updates. " \
                 "If there is any updates in the sequence, incorrect ones could include incorrect updates or mentions that it is the first time the user mentioned this thing or activity. " \
                 "Do NOT modify the most recent one (the right most one in the sequence). If the sequence has no preference updates, incorrect ones could flip the preference or add one additional change. " \
                 'Each option should share similar tone, matching length, and equal level of detail. Output a Python list of three strings, following this format: ["xxx", "yyy", "zzz"]. Please use double quotes for each string. No other words.'

    elif action == 'extract_object':
        prompt = "You have two tasks. First, please extract the primary noun from the following phrase, ignoring all adjectives or descriptors. Output a single word or short phrase only into the key 'parent_object':\n\n" + data + "\n\n" \
                 "Second, based on the extracted primary noun, propose one different child object name under this parent category, adding some different adjectives or descriptors. Output it into the key 'random_child_object'." \
                 "You should output a dictionary following this format:\n" \
                 "{\n" \
                 '    "parent_object": xxx,\n' \
                 '    "random_child_object": yyy\n' \
                 "}\n" \
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'extract_identity':
        prompt = "Please extract the gender and racial identities from the following persona information. Output a single string. No other words. Here is the full persona:\n\n" + data
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
                 "Do NOT modify the names of these keys. Please use DOUBLE quotes in order to generate the correct JSON format. No other words."
    elif action == 'propose_incorrect_recommendations':
        prompt = "Given the following response: " + data['model_response'] + "\n\n to the question: " + data['question'] + "\n\n" \
                 "propose two incorrect responses on purpose to prepare a multiple choice Q&A. "\
                 "Make sure that the incorrect answers are still good suggestions to other users, but just not for this specific user or VIOLATE this user's most recent preferences: " + data['preference'] + \
                 "If the user's preference is about liking something, the incorrect answer should talk about somehow opposite things, as if the model does not remember what this user's preferences are. " \
                 "If the user's preference is about disliking something, the incorrect answer should talk about things this user dislikes. " \
                 'Each option should share similar tone, matching length, and equal level of detail. Output a Python list of two strings, following this format: ["xxx", "yyy"]. Please use double quotes for each sentence. Do NOT use JSON. No other words.'
    elif action == 'propose_stereotypical_recommendation':
        prompt = "Given the following question: " + data['question'] + " and correct response " + data['model_response'] + "\n\n, " \
                 "prepare one incorrect answer that is stereotypical to this user's gender and racial identities, but irrelevant to the specific context " \
                 "and irrelevant to or violate this user's actual preference. Here is the user's identities:\n\n" + data['persona'] + "\n\n"\
                 "Follow the same length as the correct answer. Output the answer part only using a simple string, like 'xxx'. No additional words. " \
                 "Remember we are creating misleading options in a multiple choice question, so make it sounds like a correct one but do NOT mention that this is actually stereotypical. No other words."
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
                "Do NOT modify the names of these keys.  Please use double quotes for each key and value. No other words."
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