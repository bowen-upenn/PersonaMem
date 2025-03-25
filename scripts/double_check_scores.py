import os
import re
import pandas as pd

# List of files to process
file_paths = [
    "data/results/eval_results_128k_gpt-45.csv",
    # Add more file paths as needed
]

# Function to extract options considering parentheses
def extract_only_options(text):
    text = text.lower()
    in_parens = re.findall(r'\(([a-d])\)', text)
    if in_parens:
        return set(in_parens)
    else:
        return set(re.findall(r'\b([a-d])\b', text))

# Function to correct scores in a DataFrame
def correct_scores(df):
    for idx, row in df.iterrows():
        if row['score'] == False:
            correct = row['correct_answer'].lower().strip("() ")
            pred = str(row['predicted_answer'])
            response = str(row['model_response'])

            pred_options = extract_only_options(pred)
            if pred_options == {correct}:
                df.at[idx, 'score'] = True
                continue

            response_options = extract_only_options(response)
            if response_options == {correct}:
                df.at[idx, 'score'] = True
    return df

# Process and save each file with an "_updated" suffix
for path in file_paths:
    df = pd.read_csv(path)
    corrected_df = correct_scores(df)

    base, ext = os.path.splitext(path)
    updated_path = f"{base}_updated{ext}"
    corrected_df.to_csv(updated_path, index=False)

    print(f"Updated file saved to: {updated_path}")
