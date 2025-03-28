from collections import defaultdict

import numpy as np
import pandas as pd
import simpledorff
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

STATEMENT_COLS = ["s1", "s2", "s3", "s4"]


def is_odd(num):
    return num % 2


def convert_data(df):
    return_dict = defaultdict(list)
    cols = list(df.columns)
    for i, row in df.iterrows():
        item = {}
        for key_label in cols:
            key = key_label.split(":::")
            if len(key) == 1:
                item[key[0]] = row[key_label]
            elif len(key) == 2:
                statement, bool_label = key
                val = row[key_label]
                if val != val:  # check for nan
                    continue
                statement_num = statement[0:2]
                assert statement_num not in item
                # True is always odd, False always even
                item[statement_num] = is_odd(val)
        for key in item:
            return_dict[key].append(item[key])
    return pd.DataFrame(return_dict)


def majority_vote(l):
    if len(l) == 1:
        return l[0]
    d = defaultdict(int)
    for elem in l:
        d[elem] += 1
        if d[elem] > len(l) / 2:
            return elem
    # if there is a tie, return the key with the highest value
    return max(d.keys())


PATH = "annotation_output/full/annotated_instances.tsv"
raw_df = pd.read_csv(PATH, sep="\t", on_bad_lines="warn")
df = convert_data(raw_df)

agg_df = (
    df.groupby("instance_id")
    .agg(
        {
            "displayed_text": "first",  # Assuming displayed_text is identical
            "s1": list,
            "s2": list,
            "s3": list,
            "s4": list,
            "user": list,
        }
    )
    .reset_index()
)
agg_df["num_users"] = agg_df["user"].apply(len)

all_users = sorted(df["user"].unique())


def get_ordered_annotations(row):
    annotations = {col: [] for col in STATEMENT_COLS}
    user_response_map = {
        user: [row[col][i] for col in STATEMENT_COLS] for i, user in enumerate(row["user"])
    }
    for user in all_users:
        for col_idx, col in enumerate(STATEMENT_COLS):
            if user in user_response_map:
                annotations[col].append(user_response_map[user][col_idx])
            else:
                annotations[col].append(1)  # Missing data
    return annotations


agg_df["annotations"] = agg_df.apply(get_ordered_annotations, axis=1)

print("OVERALL STATISTICS")
for statement in STATEMENT_COLS:
    statement_maj = f"{statement}_maj"
    agg_df[statement_maj] = agg_df[statement].map(majority_vote)
    mean = agg_df[statement_maj].mean()
    print(f"  {statement} mean: {mean:.1%}")

multi_df = agg_df[agg_df["num_users"] > 1]

for user in all_users:
    print(user)
    user_df = df[df["user"] == user]
    for statement in STATEMENT_COLS:
        col = user_df[statement]
        print(f"  {statement} mean: {col.mean():.1%}")

print("krippendorff's alpha")

for statement in STATEMENT_COLS:
    alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
        df, experiment_col="instance_id", annotator_col="user", class_col=statement
    )
    print(f"  {statement}: {alpha}")
