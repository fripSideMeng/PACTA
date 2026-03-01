import pandas as pd
import random
import numpy as np
import re
from collections import Counter
from src.peft_sampling.sample_peft_train_dist import sample_df
from src.data import numeric_labels, fix_labels,\
                     always_numeric_labels, context_labels


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
    all_labels = set([fix_labels(s, lsd) for s in lsd['label_set']])
    num_labs = set([fix_labels(s, lsd) for s in always_numeric_labels])
    templates = ['INSTRUCTION: From the following list of labels, choose one '\
                 + 'that best describes the column.\nLabels: {}.\nColumn: '\
                 + '{}\nAnswer: \n', 'Column: {}\nWhich of the following labels '\
                 + 'matches best?\n{}', 'Select the option which best describes '\
                 + 'the input.\nINPUT: {}.\nOPTIONS:\n{}\n']
    cache: dict[str, pd.DataFrame] = dict()
    # labels_dist = Counter([fix_labels(s, lsd)
    #                        for s in label_df['label'].tolist()])

    for row in label_df.itertuples(index=False):
        tn = row.table_name
        col = row.column_index
        if tn not in cache:
            df = pd.read_json(f"sotab91_train/Train/{tn}", lines=True, 
                              compression="gzip").astype(str)
            cache[tn] = df
        else:
            df = cache[tn]
        series = df[col].explode().dropna()
        is_numeric = is_numeric_col(series)
        gt_label = str(row.label)
        if gt_label in lsd['dict_map']:
            gt_label = lsd['dict_map'][gt_label]
        else:
            gt_label = gt_label.lower()

        sample_aug_list: list[list[str]] = list()
        sample_list: set[str] = set()
        for p in sorted(pd.unique(series).tolist()):
            if p in ignore_list:
                continue
            sample_list.add(p)
        if not sample_list:
            continue
        sample_list: list[str] = sorted(list(sample_list), key=len)
        sample_aug_list.append(random.choices(sample_list, k=3))
        sample_aug_list.append([ele for ele in sample_list[:3]])
        sample_list.reverse()

        weights = np.linspace(1, 0.1, len(sample_list))
        weights = weights / np.sum(weights)
        ss_orig = max_tokens // (sample_size * 3)

        if len(sample_list) > ss_orig:
            sample_list = np.array(sample_list)
            indices = np.random.choice(np.arange(len(sample_list)), 
                                        size=ss_orig, replace=False, p=weights)
            sample_list = sample_list[indices].tolist()
        if not sample_list:
            sample_list = ['None']
        if len(sample_list) < sample_size:
            sample_list = sample_list * sample_size
        if len(sample_list) > sample_size:
            sample_list = sample_list[:sample_size]
        assert len(sample_list) == sample_size, "An index in val_indices is length "\
                                                + str(len(sample_list))
        cxt_labels = numeric_labels if is_numeric\
                     else sorted(list(all_labels.difference(num_labs)))
        lb = '\n'.join(['- ' + cl for cl in cxt_labels])
        sample_aug_list.insert(0, sample_list)

        for t_idx, template in enumerate(templates):
            sample_aug = sample_aug_list[t_idx]
            entries = "[" + ", ".join(sample_aug).replace("[", "")\
                      .replace("]", "").replace("'", "") + "]"
            if t_idx == 0:
                s = template.format(lb, entries)
            else:
                s = template.format(entries, lb)
            descriptions.append({'input': s, 'label': gt_label})

    return descriptions


if __name__ == "__main__":
    for seed in [42, 1902582]:
        for max_length in[512, 2048]:
            for portion in [0.005, 0.01, 0.02, 0.33]:
                if max_length == 512 and portion != 0.33 and seed == 1902582:
                    continue
                label_df = pd.read_csv(f"sotab91_train/CTA_training_gt.csv")
                label_df = sample_df(label_df, portion, seed)
                print(label_df.shape)
                assert len(set(label_df['label'].tolist())) == 91
                print("Tables ready for processing...")
                pd.to_pickle(process_dataframes(label_df, 5, seed, max_length // 2), 
                            f"sotab91_train/{seed}/td_{portion}_{max_length}_aug.pkl")