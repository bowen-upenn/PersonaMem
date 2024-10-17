import json
import utils

# Sample JSON content for testing
json_content = {
    "Initial Conversation": [
        "Therapist: Hi, thanks for coming today. What would you like to discuss?",
        "Patient: Hi, thank you for seeing me. I have been feeling overwhelmed and stressed due to my work as a guesthouse owner.",
        "Therapist: It sounds like your work is taking a toll on you. Can you tell me more about what's been happening?",
        "Side_Note: Opened the guesthouse with a focus on offering a charming and cozy atmosphere. 11/15/2013",
        "Patient: I opened my guesthouse about ten years ago because I wanted to create a cozy and charming atmosphere for my guests.",
        "Patient: Ever since, I've been passionate about giving exceptional service, but lately, managing everything has been tough.",
        "Therapist: It's great that you're dedicated to your work, but sometimes these expectations can become burdensome."
    ],
    "Expanded Conversation": [
        "Therapist: It's good to see you again. How have things been since our last session?",
        "Patient: A lot has changed. I decided to close my guesthouse for a month to focus on myself and reflect.",
        "Side_Note: Decided to close the guesthouse for a month for personal rejuvenation and reflection. 01/15/2024",
        "Patient: I've also started taking pottery classes. It's been a great creative outlet and stress relief.",
        "Side_Note: Started attending pottery classes as a creative outlet and stress relief. 02/01/2024",
        "Therapist: That sounds wonderful. How are you finding the balance between creativity and work?",
        "Patient: It's been refreshing to have something that's just for me, without the pressure of perfection."
    ],
    "Other Data": "This is some other meta data that is not part of the conversation."
}

def extract_conversation(json_content, context, which_conversation='all', which_format='string'):
    """
    :param json_content: it is the JSON file that contains the conversational data
    :param context: context of the conversation, like therapy
    :param which_conversation: either "Initial", "Expanded", or "All"
    :param which_format: either string or api_dict, where api_dict is a list of dictionaries with role and content keys for LLM API
    """
    # Extracting the conversation from the JSON dictionary
    initial_conversation = json_content.get("Initial Conversation", [])
    extended_conversation = json_content.get("Expanded Conversation", [])

    if which_conversation.lower() == 'initial':
        target_conversation = initial_conversation
    elif which_conversation.lower() == 'expanded':
        target_conversation = extended_conversation
    elif which_conversation.lower() == 'all':
        target_conversation = initial_conversation + extended_conversation
    else:
        raise ValueError("Invalid conversation type: choose 'Initial', 'Expanded', or 'All'.")

    if which_format == 'string':
        # Format as a pure string, removing lines that start with 'Side_Note'
        extracted_conversation = "\n".join([line for line in target_conversation if not line.startswith("Side_Note")])
    elif which_format == 'api_dict':
        # Format the list for an LLM API in a message format
        extracted_conversation = []
        for line in target_conversation:
            if not line.startswith("Side_Note"):
                if context == 'therapist':
                    if not line.startswith("Side_Note"):
                        role = "assistant" if line.startswith("Therapist") else "user"
                        extracted_conversation.append({"role": role, "content": line})
                else:
                    raise NotImplementedError("Unknown context: {}".format(context))
    else:
        raise NotImplementedError("Unknown format: {}".format(which_format))

    return extracted_conversation


if __name__ == '__main__':
    # Run test cases on the toy example
    possible_which_conversation = ['Initial', 'Expanded', 'All']
    possible_which_format = ['string', 'api_dict']

    for which_conversation in possible_which_conversation:
        for which_format in possible_which_format:
            extracted_conversation = extract_conversation(json_content, context='therapist', which_conversation=which_conversation, which_format=which_format)
            print(f'{utils.Colors.OKGREEN}Conversation type: {which_conversation}, Output format: {which_format}{utils.Colors.ENDC}')
            print(extracted_conversation)
            print('\n')