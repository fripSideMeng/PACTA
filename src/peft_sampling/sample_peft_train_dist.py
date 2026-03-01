import pandas as pd
import numpy as np
import random
import math
import re
from src.data import context_labels, fix_labels


def is_numeric_col(col: pd.Series):
    return pd.api.types.is_numeric_dtype(col) or\
           all([re.sub('[\W_]+', '', s).isdigit() 
                for s in col.tolist()])


def process_dataframes(label_df: pd.DataFrame,
                       sample_size: int,
                       rand_seed: int,
                       max_tokens: int):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    descriptions: list[dict[str, str]] = list()
    ignore_list = ["None", 'none', 'NaN', 'nan', 'N/A', 'na', '']
    lsd = context_labels
    template = 'INSTRUCTION: From the following list of labels, choose one '\
                + 'that best describes the column.\nLabels: {}.\nColumn: '\
                + '{}\nAnswer: \n'
    cache: dict[str, pd.DataFrame] = dict()

    for row in label_df.itertuples(index=False):
        tn = row.table_name
        col = row.column_index
        if tn not in cache:
            df = pd.read_json(f"sotab91_train/Train/{tn}",
                              lines=True, compression="gzip")
            cache[tn] = df
        else:
            df = cache[tn]
        series = df.astype(str)[col].explode().dropna()
        is_numeric = is_numeric_col(series)
        gt_label = str(row.label)
        if gt_label in lsd['dict_map']:
            gt_label = lsd['dict_map'][gt_label]
        else:
            gt_label = gt_label.lower()

        sample_list = []
        for p in sorted(pd.unique(series).tolist()):
            if p in sample_list or p in ignore_list:
                continue
            sample_list.append(p)
        sample_list = sorted(sample_list, key=len, reverse=True)

        weights = np.linspace(1, 0.1, len(sample_list))
        weights = weights / np.sum(weights)
        ss_orig = max_tokens // (sample_size * 3)

        if len(sample_list) > ss_orig:
            sample_list = np.array(sample_list)
            indices = np.random.choice(np.arange(len(sample_list)), 
                                        size=ss_orig, replace=False, p=weights)
            sample_list = sample_list[indices].tolist()
        if not sample_list:
            sample_list = ["None"]
        if len(sample_list) < sample_size:
            sample_list = sample_list * sample_size
        if len(sample_list) > sample_size:
            sample_list = sample_list[:sample_size]
        assert len(sample_list) == sample_size, "An index in val_indices is length "\
                                                + str(len(sample_list))
        
        all_labels = set([fix_labels(s, lsd) for s in lsd['label_set']])
        cxt_labels = sorted(list(all_labels))
        lb = '\n'.join(['- ' + cl for cl in cxt_labels])
        entries = "[" + ", ".join(sample_list).replace("[", "")\
                  .replace("]", "").replace("'", "") + "]"
        s = template.format(lb, entries)
        descriptions.append({'input': s, 'label': gt_label})

    return descriptions


def sample_df_ks(df: pd.DataFrame, k: int, seed: int):
    labels = set(df['label'].tolist())
    assert len(labels) == 91
    sub_dfs = list()
    for label in labels:
        sub_df = df[df['label'] == label]
        sub_df = sub_df.sample(n=k, random_state=seed)
        sub_dfs.append(sub_df)
    return pd.concat(sub_dfs)


def sample_df(df: pd.DataFrame, ratio: float, seed: int):
    labels = set(df['label'].tolist())
    assert len(labels) == 91
    sub_dfs = list()
    for label in labels:
        sub_df = df[df['label'] == label]
        nums = max(math.floor(sub_df.shape[0] * ratio), 1)
        sub_df = sub_df.sample(n=nums, random_state=seed)
        sub_dfs.append(sub_df)
    return pd.concat(sub_dfs)


if __name__ == "__main__":
    for seed in [42, 1902582]:
        for max_length in [512, 2048]:
            for kshot in [1, 3, 5]:
                label_df = pd.read_csv(f"sotab91_train/CTA_training_gt.csv")
                label_df = sample_df_ks(label_df, kshot, seed)
                print(label_df.shape)
                assert len(set(label_df['label'].tolist())) == 91
                print("Tables ready for processing...")
                pd.to_pickle(process_dataframes(label_df, 5, seed, max_length // 2), 
                             f"sotab91_train/{seed}/td_{kshot}shot_{max_length}.pkl")
