from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,\
                         DataCollatorForSeq2Seq, set_seed, T5ForConditionalGeneration
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from peft import get_peft_model, TaskType, LoraConfig,\
                 PromptTuningConfig, PromptTuningInit
from datasets import Dataset
from functools import partial
from sklearn.metrics import f1_score
from argparse import Namespace, ArgumentParser
import pandas as pd
import numpy as np
import random
import torch
import os
import re


class RandomPromptCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        for feat in features:
            if isinstance(feat['input_ids'][0], list):
                chosen_idx = random.randint(0, 2)
                feat['input_ids'] = feat['input_ids'][chosen_idx]
                feat['attention_mask'] = feat['attention_mask'][chosen_idx]
        return super().__call__(features, return_tensors=return_tensors)


def preprocess(example, tokenizer,
               max_token_length, ft_type='lora'):
    input_text = example['input']
    input_enc = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_token_length
    )
    label_enc = tokenizer(
        example["label"],
        truncation=True,
        max_length=max_token_length
    )
    labels = [l if l != tokenizer.pad_token_id else -100
              for l in label_enc["input_ids"]]

    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": labels
    }


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_str = [p.strip().lower() for p in pred_str]
    label_str = [l.strip().lower() for l in label_str]
    f1 = f1_score(pred_str, label_str,
                  average="micro", zero_division=0)
    # accuracy = sum(p == l for p, l in zip(pred_str, label_str)) / len(pred_str)

    return {"f1": round(f1, 6)}


def run(args: Namespace):
    max_length = 2048 if args.model_name == "ul2" else 512
    aug_suffix = "_aug" if args.augment else ""
    input_td = os.path.join(f"{args.ds}_train", str(args.rand_seed),
                            f"td_{args.tr_ratio}_{max_length}{aug_suffix}.pkl")
    input_vd = os.path.join(f"{args.ds}_val", str(args.rand_seed),
                            f"vd_{max_length}_dist.pkl")
    train_data: list[dict[str, str]] = pd.read_pickle(input_td)
    if args.ablate >= 0:
        assert args.sample == -1 and args.augment
        train_data = [ele for tidx, ele in enumerate(train_data)
                      if tidx % 3 != args.ablate]
    if args.sample >= 0:
        assert args.ablate == -1 and args.augment
        patterns = [r"(Column: (.*?)\nAnswer)", r"(Column: (.*?)\nWhich)",
                    r"(INPUT: (.*?)\.\nOPTIONS)"]
        ns = ["Column: {}\nAnswer", "Column: {}\nWhich", "INPUT: {}.\nOPTIONS"]
        other_two = [tuple(i for i in range(3) if i != j) for j in range(3)]
        for td_idx in range(0, len(train_data), 3):
            dc = re.search(patterns[args.sample], train_data[td_idx + args.sample]['input'],
                           re.DOTALL).group(2)
            fms = [re.search(patterns[tpidx], train_data[td_idx + tpidx]['input'],
                             re.DOTALL).group(1) for tpidx in other_two[args.sample]]
            for tpidxidx, tpidx in enumerate(other_two[args.sample]):
                text = train_data[td_idx + tpidx]['input']
                train_data[td_idx + tpidx]['input'] = text.replace(fms[tpidxidx],
                                                                   ns[tpidx].format(dc))

    val_data: list[dict[str, str]] = pd.read_pickle(input_vd)
    dataset = Dataset.from_list(train_data)
    dataset_val = Dataset.from_list(val_data)

    mp = f"flan-{args.model_name}"
    mp = os.path.join(os.environ["SLURM_TMPDIR"], mp)
    tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="right")
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "t5" in args.model_name:
        dtype = torch.bfloat16 if args.model_name == "t5-xxl" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(mp, torch_dtype=dtype,
                                                      device_map="auto",
                                                      low_cpu_mem_usage=True)
    else:
        model = T5ForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.bfloat16,
                                                           device_map="auto",
                                                           low_cpu_mem_usage=True)
    # model.eval()

    aug_peft = f"-{args.ft_type}-aug" if args.augment else f"-{args.ft_type}"
    ablate = f"-ablate{args.ablate}" if args.ablate >= 0 else ''
    sample = f"-sample{args.sample}" if args.sample >= 0 else ''
    peft_path = f"peft-flan-{args.model_name}-classifier-{args.tr_ratio}{ablate}{sample}{aug_peft}"
    peft_path = os.path.join(os.environ["HOME"], "scratch",
                             args.ds, str(args.rand_seed), peft_path)
    os.makedirs(peft_path, exist_ok=True)

    data_collator = RandomPromptCollator(tokenizer=tokenizer, model=model,
                                         padding="longest", return_tensors="pt")

    if args.ft_type != "full":
        if args.ft_type == "lora":
            r = 16
            lora_alpha = 2 * r
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q", "k", "v", "o"],
                use_rslora=True
            )
        else:
            peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                num_virtual_tokens=50,
                prompt_tuning_init=PromptTuningInit.RANDOM,
                tokenizer_name_or_path=mp,
                base_model_name_or_path=mp
            )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenized_dataset = dataset.map(partial(preprocess, tokenizer=tokenizer,
                                            max_token_length=max_length,
                                            ft_type=args.ft_type),
                                    num_proc=1)
    tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)
    tokenized_dataset_val = dataset_val.map(partial(preprocess, tokenizer=tokenizer,
                                                    max_token_length=max_length),
                                            num_proc=1)
    tokenized_dataset_val = tokenized_dataset_val.remove_columns(dataset_val.column_names)

    # if len(dataset) >= 100000:
    #     actual_batch_size = 32
    if len(dataset) >= 40000:
        actual_batch_size = 32
    elif len(dataset) >= 3000:
        actual_batch_size = 16
    elif len(dataset) >= 1000:
        actual_batch_size = 8
    else:
        actual_batch_size = 4

    if args.model_name == "ul2":
        tr_batch_size = 1
    elif args.model_name == "t5-xxl":
        tr_batch_size = 2
    else:
        tr_batch_size = actual_batch_size

    if args.model_name == "t5-base":
        eval_batch_size = 32
    else:
        eval_batch_size = 4

    gr_accum_steps = (actual_batch_size // tr_batch_size)

    if len(dataset) >= 100000:
        logging_steps = 5000
    elif len(dataset) >= 40000:
        logging_steps = 2500
    elif len(dataset) >= 3000:
        logging_steps = 500
    else:
        logging_steps = 100

    if len(dataset) >= 20000:
        if args.model_name == "t5-base":
            num_epochs = 8
        else:
            num_epochs = 3
    else:
        num_epochs = 10

    if args.ft_type != "full":
        if args.ft_type == "lora":
            if actual_batch_size <= 8:
                lr = 1e-4
            elif actual_batch_size <= 32:
                lr = 2e-4
            else:
                lr = 4e-4
        else:
            lr = 0.3
    else:
        if actual_batch_size <= 8:
            lr = 1e-5
        elif actual_batch_size <= 32:
            lr = 2e-5
        else:
            lr = 4e-5
    bf16 = (args.model_name != "t5-base")

    training_args = Seq2SeqTrainingArguments(
        output_dir=peft_path,
        per_device_train_batch_size=tr_batch_size,
        gradient_accumulation_steps=gr_accum_steps,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        per_device_eval_batch_size=eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none",
        label_names=["labels"],
        predict_with_generate=True,
        generation_max_length=32,
        dataloader_pin_memory=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics,
                                tokenizer=tokenizer)
    )

    print("Training starts...")
    trainer.train()
    tokenizer.save_pretrained(peft_path)


def main():
    parser = ArgumentParser(description="Takes input specs for the 'run' function.")

    parser.add_argument("--model_name", type=str,
                        help="Model name: supported models",
                        required=True, choices=["ul2", "t5-xxl", "t5-base"])
    parser.add_argument("--ablate", type=int,
                        help="Ablate the template number",
                        default=-1)
    parser.add_argument("--sample", type=int,
                        help="sample method",
                        default=-1)
    parser.add_argument("--rand_seed", type=int,
                        help="Random seed",
                        default=1902582)
    parser.add_argument("--tr_ratio", type=float,
                        help="Ratio of the original training data",
                        default=0.0)
    parser.add_argument("--ft_type", type=str, help="PEFT method",
                        required=True, choices=["lora", "spt", "full"])
    parser.add_argument("--augment", action='store_true',
                        help="Use data augmentation")
    parser.add_argument("--ds", type=str, help="Dataset",
                        default="sotab91", choices=["viznet", "sotab91"])

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.rand_seed)
    run(args)


if __name__ == "__main__":
    main()
