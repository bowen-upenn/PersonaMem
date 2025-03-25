import ast
import json

import pandas as pd


def read_jsonl(PATH):
    data = []
    with open(PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    dict_output = {}
    for block in data:
        keys = list(block.keys())
        if len(keys) != 1:
            raise ValueError(f"Expected a single key in the dictionary, got {len(keys)} keys.")
        dict_output[keys[0]] = [json.dumps(c) for c in block[keys[0]]]

    return dict_output


def data_loader(PATH_questions, PATH_contexts, fix_json=False):
    # This function loads the questions and contexts from the given paths
    # If fix_json is True, it will fix the json file at PATH_contexts
    # by replacing the '}{\n' with a comma
    #
    # Args:
    # PATH_questions: str, path to the questions file
    # PATH_contexts: str, path to the contexts file
    # fix_json: bool, whether to fix the json file at PATH_contexts
    #
    # Returns:
    # questions: pd.DataFrame, questions data
    # contexts: dict, contexts data

    # # Load the data
    # if fix_json:
    #     with open(PATH_contexts, 'r') as file:
    #         lines = file.readlines()

    #     # Process lines
    #     new_lines = []
    #     for i, line in enumerate(lines):
    #         if line == "}{\n":
    #             # Replace the last '\n' in the previous line with a comma
    #             if new_lines:
    #                 new_lines[-1] = new_lines[-1].rstrip('\n') + ','
    #             # Skip adding this line (effectively removing it)
    #         else:
    #             new_lines.append(line)

    #     # Write the modified content back to the file
    #     with open(PATH_contexts, 'w') as file:
    #         file.writelines(new_lines)

    questions = pd.read_csv(PATH_questions)
    contexts = {}
    if PATH_contexts:
        contexts = read_jsonl(PATH_contexts)

    # preprocessing
    questions["all_options"] = questions["all_options"].apply(lambda x: ast.literal_eval(x))

    return questions, contexts


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si : si + (d + 1 if i < r else d)]
