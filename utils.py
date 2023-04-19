import torch
import ray
import pandas as pd
import numpy as np

from functools import partial
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomModel

DATASET_FOLDER = "./"
DEFAULT_SEED = 1234

def compute_metrics(eval_preds, tokenizer):
    """
    Remember to use AutoModel for generation here
    """

    import evaluate
    import nltk
    import numpy as np
    from nltk.tokenize import sent_tokenize
    nltk.download("punkt")

    # Metric
    metric = evaluate.load("rouge")

    # helper function to postprocess text
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def preprocess_function(samples, tokenizer, max_length):
    
    infos = samples["info"]
    summaries = samples["summaries"]
    choices = samples["choice"]

    anchors = ["summarize: " + info["post"] for info in infos]
    positives = []
    negatives = []

    for i, values in enumerate(summaries):
        positives.append(values[choices[i]]["text"])
        negatives.append(values[choices[i]-1]["text"])

    anch_tokens = tokenizer(anchors, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    pos_tokens = tokenizer(positives, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    neg_tokens = tokenizer(negatives, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")

    model_inputs = {
        "anchor": anch_tokens.input_ids,
        "anchor_mask": anch_tokens.attention_mask,
        "positive": pos_tokens.input_ids,
        "positive_mask": pos_tokens.attention_mask,
        "negative": neg_tokens.input_ids,
        "negative_mask": neg_tokens.attention_mask
    }

    return model_inputs


def preprocess_dataset(dataset, tokenizer, max_length):

    _preprocessing_function = partial(preprocess_function, tokenizer=tokenizer, max_length=max_length)

    return dataset.map(
        _preprocessing_function, batched=True
    ).remove_columns(["info", "split", "summaries", "choice", "worker", "batch", "extra"])

def load_training_dataset(tokenizer, max_length):
    if not ray.is_initialized():
        ray.init()

    raw_dataset = Dataset.from_pandas(ray.data.read_json(f"{DATASET_FOLDER}/tldr_train.json").to_pandas())
    train_dataset = preprocess_dataset(raw_dataset, tokenizer, max_length)

    return train_dataset

def load_evaluation_dataset(tokenizer, max_length):
    raw_dataset = load_dataset("json", data_files={"validation": f"{DATASET_FOLDER}/tldr_validation.json"})["validation"]
    eval_dataset = train_dataset = preprocess_dataset(raw_dataset, tokenizer, max_length).remove_columns(["negative", "negative_mask", "positive_mask"])
    eval_dataset = eval_dataset.rename_column("anchor", "input_ids").rename_column("positive", "labels").rename_column("anchor_mask", "attention_mask")
    return eval_dataset

def load_model(checkpoint, base_model):
    if base_model:
        model = BloomModel.from_pretrained(checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)

    return model

def load_tokenizer(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_proc=4, use_fast=True)
    return tokenizer

def get_model_tokenizer(checkpoint, device, base_model=False):
    model = load_model(checkpoint, base_model).to(device)
    tokenizer = load_tokenizer(checkpoint)
    return model, tokenizer
    