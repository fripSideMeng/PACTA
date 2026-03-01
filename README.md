# PACTA: Prompt Augmentation for Column Type Annotation

This repository contains the implementation for **"PACTA: Prompt Augmentation for Column Type Annotation"**, including the training script, environment setup, and released LoRA adapter weights for the [SOTAB-91](https://webdatacommons.org/structureddata/sotab/) and [VizNet](https://github.com/mitmedialab/sherlock-project) benchmarks.

---

## Overview

This project addresses **prompt sensitivity** in large language model (LLM) fine-tuning for the **Column Type Annotation (CTA)** task, which plays a central role in data integration and semantic data curation.  
We propose a **prompt augmentation strategy** within a **parameter-efficient fine-tuning (LoRA)** framework to improve robustness across prompt variations and datasets.

---

## Environment Setup

To set up the environment, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Training Usage
The repository contains a single trainer script, peft_flan.py, which defines the training class and fine-tuning workflow.

To call it for fine-tuning, use:

```bash
python peft_flan.py --model_name <model> --ft_type lora --augment --tr_ratio <training data ratio>
```
The `model` can be one of `t5-base`, `t5-xxl`, or `ul2`.

Ensure that the training data is located under the `<dataset>_<split>/<rand_seed>` folder, where:
- `<dataset>` is either `sotab91` or `viznet`
- `<split>` is either `train` or `val`
- `<rand_seed>` is the specified random seed

---

## Inference
To perform inference after training, navigate to the parent directory of the `src/` folder (which contains the source code), and run the following command:

```bash
python -m src.run \
  --model_name=<model> \
  --save_path=<path> \
  --input_files=<input files> \
  --input_labels=<label_set_csv_file> \
  --label_set=<label_set_name> \
  --method <methods> \
  --tr_ratio <labeled_ratio> \
  --peft_augment \
  --results \
  --response \
  --rules True \
  --rand_seed <seed>
```

This command follows the implementation from [ArcheType](https://github.com/penfever/ArcheType), with added logic for loading and using trained LoRA adapters during inference.

- `<model>` specifies the model to be loaded (see [`src/model.py`](src/model.py) for available model names).
- `--tr_ratio` defines the ratio of the full training data used during fine-tuning.
- `--peft_augment` indicates whether the model was fine-tuned with prompt-augmented training data.
- All other parameters can be configured as described in the [ArcheType documentation](https://github.com/penfever/ArcheType).

---

## Released LoRA Adapters & Datasets
LoRA adapter weights trained on both benchmarks (with 2% labeled data and random seed of 1902582) are [publicly available](https://drive.google.com/drive/folders/1VgpeZCcvCESzCaTPD1kzWFxw2CF3CY6q?usp=sharing). See the [`src/peft_sampling`](src/peft_sampling) for train/validation datasets creation.
