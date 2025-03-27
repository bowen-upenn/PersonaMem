import argparse
from pathlib import Path

import pandas as pd

# calculate classification_report with sklearn
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--input_paths", "-i", type=Path, nargs="+")
parser.add_argument("--output_path", "-o", type=Path)


if __name__ == "__main__":
    args = parser.parse_args()

    predictions = []
    gold_labels = []
    dfs = []
    for input_path in args.input_paths:
        df = pd.read_csv(input_path)
        predictions.extend(df["p"])
        gold_labels.extend(df["answer"])
        dfs.append(df)

    print(classification_report(gold_labels, predictions, digits=3))
    if len(args.input_paths) > 1:
        stem = args.input_paths[0].stem.rsplit("_", 1)[0]
        out_path = args.output_path or args.input_paths[0].parent / f"{stem}_concat.csv"
        df_concat = pd.concat(dfs)
        df_concat.to_csv(out_path, index=False)
        print(f"Concatenated predictions with saved to {out_path}")
