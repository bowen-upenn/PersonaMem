import argparse
import json
from copy import copy
from pathlib import Path

import pandas as pd
import random

# DEFAULT_CONVO_PATH = Path("data/output/therapy/conversation_therapy_persona0_sample0.json")
CONVO_DIR = Path("data/output/")
OUT_PATH = Path("qa_annotation/data_files/personamem_annot_sample.csv")
TO_SKIP = set(
    [
        "How_Many_Pref_Updates",
        "Where",
        "Conversation",
        "other_previously_mentioned_events",
        "Incorrect_Responses",
    ]
)
AT_END = ["Type", "Question/Message", "Correct_Response"]
AT_END_SET = set(AT_END)
SEED = 2557
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", "-i", type=Path, default=CONVO_DIR)
parser.add_argument("--num_tasks_per_topic", "-n", type=int, default=-1)
parser.add_argument("--persona", "-p", type=int, default=0)


def bold(text, blue=False):
    if blue:
        return f"<span style='color:blue'><b>{text}</b></span>"
    return f"<b>{text}</b>"


def underline(text):
    return f"<u>{text}</u>"


def format_speaker(lines):
    convo_parts = []
    for line in lines:
        arr = line.split(":", 1)
        if len(arr) == 2:
            speaker, text = arr
            convo_parts.append(f"{underline(speaker)}: {text}")
    return " <br> ".join(convo_parts)


def format_for_html(entry):
    parts = []

    for key, value in entry.items():
        if key in TO_SKIP or key in AT_END_SET:
            continue
        parts.append(f"{bold(key)}: {value}")

    for key in AT_END:
        text = entry[key]
        parts.append(f"{bold(key, True)}: {text}")
    parts.append(bold("Incorrect_Responses:", True))
    for line in entry["Incorrect_Responses"]:
        parts.append(f" * {line}")
    col1 = " <br> ".join(parts)

    if "Conversation" not in entry:
        return col1

    # column 2
    parts = []
    if "other_previously_mentioned_events" in entry:
        events = format_speaker(entry["other_previously_mentioned_events"].split("\n"))
        parts.append(f"{bold('other_previously_mentioned_events')}: {events}")
    convo = format_speaker(entry["Conversation"].split("\n"))

    parts.append(f"{bold('Conversation', True)}: {convo}")
    col2 = " <br><br> ".join(parts)

    return f"""
<table>
  <tr>
    <td>{col1}</td>
    <td>{col2}</td>
  </tr>
</table>
"""


if __name__ == "__main__":
    args = parser.parse_args()

    entries_formatted = []
    for path in CONVO_DIR.glob(f"*/*persona{args.persona}_sample0.json"):
        category = path.parent.name
        print(f"Processing {path}")

        with path.open() as f:
            data = json.load(f)

        if category in set(["writing", "email", "coding"]):
            qa_data = data["Q&A"]["Conversation"]
            for i, entry in enumerate(qa_data):
                entry["id"] = f"{category}_persona{args.persona}_q{i}"
        else:
            qa_data = []
            for time_period in ["Init Conversation", "Conversation Next Week", "Conversation Next Month", "Conversation Next Year"]:
                qa_data_ = data["Q&A"][time_period]
                time_short = time_period.split()[-1] if time_period != "Init Conversation" else "Init"
                for i, entry in enumerate(qa_data_):
                    entry["id"] = f"{category}_persona{args.persona}_{time_short}_q{i}"
                qa_data += qa_data_

        num_tasks = args.num_tasks_per_topic
        if num_tasks == -1:
            num_tasks = len(qa_data)
        else:
            random.shuffle(qa_data)

        print(f"Generating tasks for {num_tasks}/{len(qa_data)} entries")
        for i, entry in enumerate(qa_data[:num_tasks]):
            entry = copy(entry)
            if isinstance(entry["Reference"], dict):
                for key, value in entry["Reference"].items():
                    entry[key] = value
                del entry["Reference"]
            entry["Question/Message"] = entry.pop("Question")
            entry['Correct_Response'] = entry.pop("Correct_Answer")
            entry['Incorrect_Responses'] = entry.pop("Incorrect_Answers")
            html_text = format_for_html(entry)
            entries_formatted.append({"id": entry['id'], "text": html_text})

    df = pd.DataFrame(entries_formatted)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} lines to {OUT_PATH}")
