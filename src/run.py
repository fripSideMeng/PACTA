import os, json, requests, copy
import collections
import hashlib
import random
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pathlib import Path
import argparse
import openai
from transformers import AutoTokenizer
import numpy as np

from src.model import init_model, get_sent_model, get_model_resp, seed_all
from src.data import get_df_sample, fix_labels, insert_source, get_lsd, get_d4_dfs,\
                     pd_read_any, get_amstr_classname_map, get_pubchem_dfs,\
                     get_pubchem_classname_map, get_viznet_classname_map, get_sample_prefix
from src.metrics import results_checker, results_checker_doduo
from src.const import DOTENV_PATH, MAX_LEN

def get_dataset_flags(model_name, args, label_set):
  #DATASET SPECIFIC FLAGS
  args['infmods'] = "sherlock" in model_name or "doduo" in model_name
  args['isd4'] = "d4" in label_set['name']
  args['isAmstr'] = "amstr" in label_set['name']
  args['isPubchem'] = "pubchem" in label_set['name']
  args['isT2D'] = "T2D" in label_set['name']
  args['isEF'] = "EF" in label_set['name']
  args['isVC'] = "viznet-chorus" in label_set['name']
  args['isViznet'] = "viznet" in label_set['name']
  args['isCustom'] = "custom" in label_set['name']
  args['isSOTAB'] = "context_labels" in label_set['name']
  return args

def get_prompt_dict(args, resume, save_path):
  args["current_prompt_hashes"] = collections.Counter()
  args["resume_prompt_hashes"] = collections.Counter()
  if resume and not os.path.isfile(save_path):
    print("Could not open save file. Starting from scratch...")
    prompt_dict = {}
  elif resume and os.path.isfile(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
      prompt_dict = json.load(f)
      prompt_dict_keys = list(prompt_dict.keys())
      if len(prompt_dict_keys) == 0:
        print("No prompts found in save file. Starting from scratch...")
        prompt_dict = {}
      else:
        test_entry = prompt_dict[prompt_dict_keys[0]]
        if test_entry.get("prompt_hash", -1) == -1:
          print("No prompt hashes found in the provided dict's first entry. Starting from scratch...")
          prompt_dict = {}
        else:
          ph_seen = set()
          for k, v in prompt_dict.copy().items():
            ph = v.get("prompt_hash", -1)
            ph_count = v.get("prompt_hash_count", 1)
            if ph == -1 or ph_count == -1:
              del prompt_dict[k]
            if ph in ph_seen:
              continue
            ph_seen.add(ph)
            if args["resume_prompt_hashes"][ph] == 0:
              args["resume_prompt_hashes"][ph] = ph_count
            else:
              args["resume_prompt_hashes"][ph] += ph_count
        print(f"Loaded {args['resume_prompt_hashes'].total()} entries from the save file.")
  else:
    prompt_dict = {}
  return prompt_dict, args

def define_inputs(inputs, args, rand_seed=13):
  # define inputs and labels for test datasets
  if args['isT2D'] or args['isEF']:
    origin_df = pd.read_json(inputs)
    contexts = origin_df['input'].tolist()
    inputs = []
    for c in contexts:
      inputs.append(pd.DataFrame(c.split(",")))
    labels = origin_df["output"].tolist()
  elif args['isVC']:
    input_df = input_df.sample(1000, replace=False)
    inputs = input_df['data'].tolist()
    labels = input_df['class'].tolist()
  elif isinstance(inputs, dict):
    labels = np.array(['_'.join(k.split('_')[:-1]) for k in inputs.keys()])
    inputs = [v for v in inputs.values()]
    np.random.seed(rand_seed)
    indices = np.arange(len(labels))
    np.random.shuffle(indices) 
    labels = labels[indices].tolist()
    new_inputs = []
    for i in indices:
      new_inputs.append(inputs[i])
    inputs = new_inputs
  else:
    # format is already OK
    labels = None
  return inputs, labels, args

def define_label_context(labels, idx, f, label_set, args, 
                         method, input_df=None, tn_df=None):
  if args['isd4']:
      f_df = f
      label_indices=[2]
      gt_labels = labels[idx]
  elif args['isT2D'] or args['isEF']\
       or args['isPubchem'] or args['isVC']:
      f_df = f
      label_indices=[0]
      gt_labels = labels[idx]
  elif "skip-eval" in args['input_labels']:
      f_df = f
      # f_df = pd_read_any(f)
      gt_labels = None
      label_indices = [i for i in range(len(f_df.columns))]
  else:
      if isinstance(f, str):
        tn = '_'.join(f.split('_')[ : 2])
        if args["isAmstr"]:
          gt_labels = '_'.join(f.split('_')[4:])
        else:
          gt_labels = f.split('_')[-1]
        label_indices = [0]
        f_df = tn_df[tn]
      else:
        gt_labels = input_df[input_df['table_name'] == f.name]
        label_indices = pd.unique(gt_labels['column_index']).tolist()
        f_df = pd.read_json(f, compression='gzip', lines=True)
  
  return f_df, label_indices, gt_labels

def get_current_context(f, col, gt_labels, label_indices, idx, args, 
                        method, label_set, sample_df: pd.DataFrame=None, 
                        table_src=False, sample_size=5, model_name=None):
  if args['isd4']:
    orig_label = gt_labels
  elif args['isAmstr']:
    amstr_classname_map = get_amstr_classname_map()
    orig_label = amstr_classname_map[gt_labels]
  elif args['isPubchem']:
    pubchem_classname_map = get_pubchem_classname_map()
    orig_label = pubchem_classname_map[gt_labels]
  elif args['isT2D'] or args['isEF']:
    orig_label = gt_labels
  elif args['isViznet']:
    viznet_classname_map = get_viznet_classname_map()
    orig_label = viznet_classname_map[gt_labels]
  elif "skip-eval" in args['input_labels']:
    orig_label = ""
  else:
    gt_row = gt_labels[gt_labels['column_index'] == idx]
    orig_label = gt_row['label'].item()
  label = fix_labels(orig_label, label_set)
  if "dedup" in method:
    limited_context = list(set(sample_df[col].tolist()[:sample_size]))
  else:
    limited_context = sample_df[col].tolist()[:sample_size]
  if table_src:
    if "zs" in model_name:
      context_n = insert_source(sample_df[col].tolist(), f.name, 
                                zs="zs" in model_name)
      args["table_name"] = context_n
      context = sample_df[col].tolist()
    else:
      context = insert_source(sample_df[col].tolist(), f.name, 
                              zs="zs" in model_name)
  else:
    if "dedup" in method:
      context = list(set(sample_df[col].tolist()))
    else:
      context = sample_df[col].tolist()
  if args["partial_oracle"]:
    limited_context[0] = context[0] = label
  elif args['oracle']:
    limited_context = context = [label]
  return orig_label, label, context, limited_context, args

def get_resp_update_dict(args, prompt_dict, save_path, label_set, context, label, 
                         link, response, session, model_name, limited_context, 
                         method, orig_label, f, idx, return_prompt=False, 
                         kshot="", row_context=None):

  args['kshot'] = kshot
  prompt_hash = hashlib.md5(str(context).encode('utf-8')).hexdigest()
  args["current_prompt_hashes"][prompt_hash] += 1
  if args["resume_prompt_hashes"][prompt_hash] >= args["current_prompt_hashes"][prompt_hash]:
    return prompt_dict, args

  try:
    key, ans_dict = get_model_resp(label_set, context, label, prompt_dict, link=link, 
                                   response=response, session=session, cbc=None, 
                                   model=model_name, limited_context=limited_context, 
                                   method=method, args=args, do_kshot=return_prompt,
                                   rows=row_context)
  except RuntimeError as r:
    print(r)
    try:
      key, ans_dict = get_model_resp(label_set, context, label, prompt_dict, link=link, 
                                     response=response, session=session, cbc=None, 
                                     model=model_name, limited_context=limited_context, 
                                     method=method, args=args, do_kshot=return_prompt,
                                     rows=row_context)
    except RuntimeError as r:
      with open(save_path, 'w', encoding='utf-8') as my_f:
        json.dump(prompt_dict, my_f, ensure_ascii=False, indent=4)
      raise RuntimeError(f"Unhandled RuntimeError: {r} \n Please check logs for more information.")

  if return_prompt:
    return key, {}, None

  # hash validation
  if args["current_prompt_hashes"][prompt_hash] == 1:
    ans_dict['prompt_hash'] = prompt_hash
    ans_dict['prompt_hash_count'] = 1
    ans_dict['original_label'] = orig_label
    ans_dict['file+idx'] = str(f) + "_" + str(idx)
    prompt_dict[key] = ans_dict
  else:
    for k, v in prompt_dict.items():
      if v.get("prompt_hash", -1) == prompt_hash:
        v['prompt_hash_count'] = args["current_prompt_hashes"][prompt_hash]
        break

  return prompt_dict, args, key

def run(
    model_name : str, 
    save_path : str, 
    inputs : list, 
    label_set : dict, 
    input_df : pd.DataFrame, 
    resume : bool = True, 
    results : bool = True, 
    stop_early : int = -1, 
    rand_seed : int = 13, 
    sample_size : int = 5, 
    link : str = None, 
    response : bool = True, 
    summ_stats : bool = False, 
    table_src : bool = False, 
    other_col : bool = False, 
    skip_short : bool = False, 
    min_var : int = 0, 
    method : list = ["similarity"],
    samples: dict[str, list[str]] = dict(),
    tn_df: dict[str, pd.DataFrame] = dict(),
    args : dict = dict()):
  
  args['context_is_numeric'] = False
  args['rand_seed'] = rand_seed

  args = get_dataset_flags(model_name, args, label_set)

  if model_name in ["llama-old", "sherlock"]:
    pass
  elif "gpt-4" in model_name or "gpt-3.5" in model_name:
    load_dotenv(DOTENV_PATH)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    assert openai.api_key != None, "api key did not load"
    args['tokenizer'] = AutoTokenizer.from_pretrained("openai-gpt")
    args['MAX_LEN'] = 4096
  else:
    print("Initializing model...")
    init_model(model_name, args)

  #Retrieve similarity model
  if "similarity" in method:
    get_sent_model(args)
  
  #prompt hashing and dict initialization
  prompt_dict, args = get_prompt_dict(args, resume, save_path)

  s = requests.Session()
  if "-zs" in model_name:
    args["base_model"].eval()
  
  inputs, labels, args = define_inputs(inputs, args, rand_seed)

  args_copy = copy.deepcopy(args)
  
  kshotstr = ""

  args = args_copy
  del args_copy

  # main loop
  for idx, f in tqdm(enumerate(inputs), total=len(inputs)):
    if stop_early > -1 and idx == stop_early:
      break

    # define context and label
    f_df, label_indices, gt_labels = define_label_context(labels, idx, f, label_set, 
                                                          args, method, input_df, tn_df)

    if "skip-eval-return" in args['input_labels']:
      f_df_return = f_df.copy()
    else:
      f_df_return = None

    sample_df = get_df_sample(f_df, rand_seed, label_indices, sample_size, 
                              full=summ_stats, other_col=other_col, 
                              max_len=args["MAX_LEN"], method=method, 
                              coherence_scores=None, args=args)
    
    f_df_cols = f_df.columns

    if "skip-eval-return" in args['input_labels']:
      f_df_return_cols = []

    for col_idx, col in enumerate(f_df_cols):
      if col_idx not in label_indices:
        continue
      

      if len(pd.unique(sample_df.astype(str)[col])) < min_var:
        continue

      if samples:
        key = f if isinstance(f, str) else f"{f.name}_{col}"
        if key in samples:
          if len(samples[key]) == 0:
            continue
          col_row_context = samples[key]
        else:
          col_row_context = None
      else:
        col_row_context = None
      
      orig_label, label, context, limited_context, args = get_current_context(f, col, gt_labels, 
                                                                              label_indices, col_idx, 
                                                                              args, method, 
                                                                              label_set, sample_df, 
                                                                              table_src, sample_size)

      prompt_dict, args, key = get_resp_update_dict(args, prompt_dict, save_path, label_set, 
                                                    context, label, link, response, s, model_name, 
                                                    limited_context, method, orig_label, f, 
                                                    col_idx, return_prompt=args["return_prompt"], 
                                                    kshot=kshotstr, row_context=col_row_context)
      

      if "skip-eval-return" in args['input_labels']:
        f_df_return_cols.append(prompt_dict[key]["original_model_answer"])

    if f_df_return is not None:
      f_df_return.columns = f_df_return_cols

  with open(save_path, 'w', encoding='utf-8') as alt_f:
    json.dump(prompt_dict, alt_f, ensure_ascii=False, indent=4)

  if args['rules'] and args['isSOTAB']:
    print("Remapping rules enabled. Applying schema match fix...")
    from src.schema_match_fix import schema_match_fix
    from src.data import get_schema_df

    schema_df = get_schema_df()
    for key in tqdm(prompt_dict.keys(), total=len(prompt_dict.keys())):
      prompt_dict[key] = schema_match_fix(prompt_dict[key], schema_df, label_set)

  with open(save_path, 'w', encoding='utf-8') as my_f:
    json.dump(prompt_dict, my_f, ensure_ascii=False, indent=4)

  # results checking
  if results:
    if "skip-eval" in args['input_labels']:
      if "skip-eval-return" in args['input_labels']:
        return f_df_return
      return None
    if model_name == "doduo":
      results_checker_doduo(save_path)
    else:
      results_checker(save_path, skip_duplicates = False, naive_score = False)

  return None

def main():
    parser = argparse.ArgumentParser(description="Takes input parameters for the 'run' function.")
    
    parser.add_argument("--model_name", type=str, help="Model name: supported models and their names are listed in init_model", required=True)
    parser.add_argument("--model_path", type=str, help="Path to ArcheType-LLAMA or zs-LLAMA model weights", default="")
    parser.add_argument("--save_path", type=str, help="Save path", required=True, default="./results/archetype_predict.json")
    parser.add_argument("--input_files", type=str, help="Path to input files", required=True)
    parser.add_argument("--label_set", type=str, help="Name of label set (SOTAB-91, SOTAB-55, SOTAB-27, D4-ZS, D4-DoDuo, amstr-ZS, pubchem-ZS, viznet-ZS, custom)", required=True)
    parser.add_argument("--custom-labels", nargs='+', type=str, help="Custom labels", required=False)
    parser.add_argument("--input_labels", type=str, help="Path to input DataFrame (CSV file) for SOTAB. skip-eval will generate predictions but will not compare them to anything. skip-eval-return will return the table with the predictions. D4 will use (internal) D4 ground-truth labels.", required=True)
    parser.add_argument("--resume", action='store_true', help="Resume")
    parser.add_argument("--results", action='store_true', help="Results")
    parser.add_argument("--stop_early", type=int, default=-1, help="Stop early")
    parser.add_argument("--rand_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_size", type=int, default=5, help="Sample size")
    parser.add_argument("--link", type=str, help="Link")
    parser.add_argument("--template", type=str, help="Zero shot prompt template",
                        default="origin", choices=["origin", "alter1", "alter2",
                                                   "origin1", "origin2", "alter3",
                                                   "alter4"])
    parser.add_argument("--response", action='store_true', 
                        help="If this flag is not called, the model will not "\
                             + "provide a prediction. No response is useful for "\
                             + "generating training datasets or testing workflows.")
    parser.add_argument("--peft_augment", action='store_true', default=False,
                        help="If this flag is not called, the peft model trained "\
                             + "with augment data will not be loaded.")
    parser.add_argument("--summ_stats", action='store_true', help="Summary statistics")
    parser.add_argument("--table_src", action='store_true', help="Table source")
    parser.add_argument("--other_col", action='store_true', help="Other column")
    parser.add_argument("--skip_short", action='store_true', help="Skip short")
    parser.add_argument("--min_var", type=int, default=0, 
                        help="Columns which contain less than min_var unique values "\
                             + "will not be evaluated")
    parser.add_argument("--method", nargs='+', type=str, default=["similarity"], 
                        help="Sets label remapping strategy. skip-existing will skip "\
                             + "columns which already have a prompt in the prompt_dict. "\
                             + "If skip-eval is in method, then no evaluation will be "\
                             + "performed. similarity will use the similarity metric to "\
                             + "find the closest label in the label set. ans_contains_gt "\
                             + "and gt_contains_ans will use contains label remapping, "\
                             + "and resample will call the LLM multiple times. If "\
                             + "check_labels is in method, then every ground truth label "\
                             + "will be verified against the values in the label set.")
    parser.add_argument("--rules", type=bool, help="Apply basic remapping rules", default=False)
    parser.add_argument("--k_shot", type=int, help="K-shot learning", default=0)
    parser.add_argument("--tr_ratio", type=float, help="training data ratio", default=0.0)
    parser.add_argument("--oracle", type=bool, help="Oracle", default=False)
    parser.add_argument("--partial_oracle", type=bool, help="Partial Oracle", default=False)

    args = parser.parse_args()
    seed_all(args.rand_seed)

    sample_instr: dict[str, tuple[tuple[str]]] = dict()
    if any(['sampling' in m for m in args.method]):
      sampling_types = '_'.join(str(args.method[-1]).split('_')[1 : -1])
      prefix = get_sample_prefix(args.label_set)
      if "flan" in args.model_name:
        if "t5" in args.model_name:
          suffix = f"sample_instr_{sampling_types}_test512.pkl"
        else:
          suffix = f"sample_instr_{sampling_types}_test2048.pkl"
      else:
        suffix = f"sample_instr_{sampling_types}_test2048.pkl"
      sample_instr_path = os.path.join(prefix, str(args.rand_seed), suffix)
      sample_instr = pd.read_pickle(sample_instr_path)

    tn_df = None
    if isinstance(args.input_files, pd.DataFrame):
      df = copy.deepcopy(args.input_files)
      input_files = [df]
      args.input_files = "skip-eval"
    elif args.input_files == "D4":
      input_files = get_d4_dfs()
    elif args.input_labels == "pubchem":
      input_files = get_pubchem_dfs(args.input_files, args.rand_seed)
    elif args.input_labels == "T2D":
      input_files = "./metadata/T2D/T2D_test_archetype_instr.json"
    elif args.input_labels == "EF":
      input_files = "./metadata/EF/EF_test_archetype_instr.json"
    elif args.input_labels == "viznet-chorus":
      input_files = [args.input_files]
    elif "viznet" in args.input_labels or\
      "amstr" in args.input_labels:
      input_files = list(sample_instr.keys())
      random.shuffle(input_files)
      if "amstr" in args.input_labels:
        tn_df = {tn: df for tn, df in pd.read_pickle(f"amstr_tables/tn_df_list.pkl")}
      else:
        tn_df = {tn: df for tn, df in pd.read_pickle(f"viznet_test/tn_df_list.pkl")}
    else:
      # Define the file extensions to search for
      extensions = ('.json', '.csv', '.json.gz', '.parquet')
      input_files = []
      for extension in extensions:
        input_files = input_files + list(Path(args.input_files)
                                         .rglob(f"**/*{extension}"))
    
    if args.input_labels == "viznet-chorus":
      input_df = pd_read_any(args.input_files)
    elif args.input_labels == "D4" or \
      "amstr" in args.input_labels or \
      "pubchem" in args.input_labels or \
      "T2D" in args.input_labels or \
      "EF" in args.input_labels or \
      "viznet" in args.input_labels or \
      "amstr" in args.input_labels or \
      "skip-eval" in args.input_labels:
      input_df = None
    else:
      input_df = pd_read_any(args.input_labels)

    if args.label_set == "custom":
      label_set = {"name" : "custom", "label_set" : args.custom_labels, 
                   "dict_map" : {c : c for c in args.custom_labels}, 
                   'abbrev_map' : {c : c for c in args.custom_labels}}
    else:
      label_set = get_lsd(args.label_set)
    arg_dict = {"MAX_LEN" : MAX_LEN, 
                "model_path" : args.model_path, 
                "lsd" : label_set, 
                "rules" : bool(args.rules), 
                "oracle" : bool(args.oracle),
                "partial_oracle" : bool(args.partial_oracle),
                "input_labels" : args.input_labels,
                "return_prompt" : False,
                "k_shot" : int(args.k_shot),
                "tr_ratio": str(args.tr_ratio),
                "peft_augment": bool(args.peft_augment),
                "template": args.template}
    if sample_instr:
      arg_dict["of"] = (sampling_types == "of")

    save_path = f"results/{args.save_path}"
    run(
        args.model_name,
        save_path,
        input_files,
        label_set,
        input_df,
        args.resume,
        args.results,
        args.stop_early,
        args.rand_seed,
        args.sample_size,
        args.link,
        args.response,
        args.summ_stats,
        args.table_src,
        args.other_col,
        args.skip_short,
        args.min_var,
        args.method,
        sample_instr,
        tn_df,
        arg_dict
    )

if __name__ == "__main__":
    main()
