import pandas as pd
import random
import math
import numpy as np
import re
from collections import Counter
from src.data import fix_labels, context_labels


def is_numeric_col(col: pd.Series):
    return pd.api.types.is_numeric_dtype(col) or\
           all([re.sub('[\W_]+', '', s).isdigit() 
                for s in col.tolist()])


def process_dataframes(tn_df: dict[str, pd.DataFrame],
                       label_df: pd.DataFrame,
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

    for row in label_df.itertuples(index=False):
        tn = row.table_name
        col = row.column_index
        df = tn_df[tn]
        series = df[col].explode().dropna()
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


def sample_by_dist(ref: Counter, label_df: pd.DataFrame, seed: int):
    ref_total = sum(ref.values())
    sub_dfs = list()
    for label in ref:
        sub_df = label_df[label_df['label'] == label]
        nums = math.ceil((ref[label] / ref_total) * label_df.shape[0])
        nums = max(math.floor(max(nums, sub_df.shape[0]) * 0.05), 1)
        sub_df = sub_df.sample(n=nums, random_state=seed)
        sub_dfs.append(sub_df)
    return pd.concat(sub_dfs)


if __name__ == "__main__":
    for max_length in [512, 2048]:
        full_df = pd.read_csv("sotab91_test/CTA_test_gt.csv")
        train_labels = Counter(full_df['label'].tolist())
        assert len(train_labels) == 91
        tables_dfs = pd.read_pickle(f"sotab91_val/tn_df_list.pkl")
        tables_dfs = {tn: df.astype(str) for tn, df in tables_dfs}
        label_df = pd.read_csv(f"sotab91_val/CTA_validation_gt.csv")
        label_df = sample_by_dist(train_labels, label_df, 42)
        print(label_df.shape)
        assert len(set(label_df['label'].tolist())) == 91
        print("Tables ready for processing...")
        pd.to_pickle(process_dataframes(tables_dfs, label_df, 
                                        5, 42, max_length // 2), 
                    f"sotab91_val/42/vd_{max_length}_dist.pkl")