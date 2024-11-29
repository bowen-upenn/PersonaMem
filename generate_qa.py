import datetime
import json
import re
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_side_notes_with_timestamps(conversation):
    """
    Extracts Side_Notes with timestamps from a conversation.
    """
    text_pattern = r'\b\[?Side[\s_]?Notes?\]?\b'
    timestamp_pattern = r"\b\d{2}/\d{2}/\d{4}\b"

    # Extract lines with 'Side_Nodes' and their timestamps
    filtered_lines = [line for line in conversation if re.search(text_pattern, line)]
    timestamps = [re.search(timestamp_pattern, line).group() for line in filtered_lines if re.search(timestamp_pattern, line)]
    return [(timestamp, line) for timestamp, line in zip(timestamps, filtered_lines)]


def find_related_data(timestamp, history_blocks):
    """
    Finds events in the provided history blocks that match the timestamp.
    """
    related_data = []
    for block in history_blocks:
        for key, value in block.items():
            if key == timestamp:
                related_data.append(value)
    return related_data


def find_most_similar_event(side_note_sentence, related_data):
    """
    The same timestamp may have multiple events, like one in the general personal history and one in the contextual one.
    This function uses SentenceBERT to locate the single event we are actually targeting.
    """
    max_similarity = -1
    most_similar_data = None

    for data in related_data:
        event_sentence = data.get("event", "")
        similarity = util.pytorch_cos_sim(
            model.encode(side_note_sentence, convert_to_tensor=True),
            model.encode(event_sentence, convert_to_tensor=True)
        )
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_data = data

    return most_similar_data


def trace_event_history(timestamp, previous_blocks):
    """
    Traces the event history recursively, if needed, for knowledge updates.
    """
    linear_graph = {}
    while True:
        event_data = None
        for block in previous_blocks:
            event_data = block.get(timestamp)
            if event_data:
                break

        if not event_data:
            break  # No further history to trace
        # print('event_data', timestamp, event_data)

        linear_graph[timestamp] = event_data
        if "Old Event" in event_data:
            # Get the timestamp of the old event
            old_event_timestamp = event_data.get("Old Event Date") or event_data.get("[Old Event Date]", "")
            # print('old_event_timestamp', old_event_timestamp)

            # Update timestamp for next iteration
            timestamp = old_event_timestamp
        else:
            break  # No further history to trace

    print("linear_graph:")
    print(json.dumps(linear_graph, indent=4))
    return linear_graph


def generate_qa_static():
    pass


def generate_qa_knowledge_update():
    pass


def process_conversation(conversation_key, data_path):
    # Load json file
    with open(data_path, 'r') as file:
        data = json.load(file)

    conversation = data.get(conversation_key, [])
    # Collect all side notes with timestamps in the current conversation
    side_notes = extract_side_notes_with_timestamps(conversation)

    history_keys = {
        "Init Conversation": ["Init General Personal History", "Init Contextual Personal History"],
        "Conversation Next Week": ["Init General Personal History", "General Personal History Next Week",
                                   "Init Contextual Personal History", "Contextual Personal History Next Week"],
        "Conversation Next Month": ["Init General Personal History", "General Personal History Next Week",
                                    "General Personal History Next Month",
                                    "Init Contextual Personal History", "Contextual Personal History Next Week",
                                    "Contextual Personal History Next Month"],
        "Conversation Next Year": ["Init General Personal History", "General Personal History Next Week",
                                   "General Personal History Next Month", "General Personal History Next Year",
                                   "Init Contextual Personal History", "Contextual Personal History Next Week",
                                   "Contextual Personal History Next Month", "Contextual Personal History Next Year"]
    }

    previous_blocks = [data.get(key, {}) for key in history_keys.get(conversation_key, [])]

    for timestamp, side_note in side_notes:
        # Find related data in the previous personal history for each current event
        related_data = find_related_data(timestamp, previous_blocks)
        if not related_data:
            continue

        # If there are more than one related data with the same timestamp, find the single correct one
        if len(related_data) > 1:
            most_similar_data = find_most_similar_event(side_note, related_data)
        else:
            most_similar_data = related_data[0]
        if not most_similar_data:
            continue

        if "Reasons of Change" in most_similar_data:
            # Knowledge update
            event_history = trace_event_history(timestamp, previous_blocks)
            # print(f"Knowledge update traced: {event_history}")
            generate_qa_knowledge_update()
        else:
            # Static knowledge point
            # print(f"Static knowledge point: {most_similar_data}")
            generate_qa_static()


if __name__ == "__main__":
    data_path = './data/output/conversation_therapy_persona0_sample0.json'
    process_conversation("Conversation Next Year", data_path)
