"""
LLM pipeline built on 🤗 Transformers.
Fine-tunes DistilBERT (or any MODEL_NAME) on a CSV in exactly the
same “key:value; …” text format you used in fine-tune.py.

Public API
----------
fine_tune_llm(dataset_path, model_name, output_root, update_progress_cb)
    → (ckpt_path, metrics_dict, confusion_matrix)

infer_llm(ckpt_path, dataset_path, output_root, update_progress_cb)
    → (metrics_dict, confusion_matrix, result_json_path)
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

# --------------------------------------------------------------------------- #
#                            CONSTANTS / PARAMS                               #
# --------------------------------------------------------------------------- #
RANDOM_SEED = 21023
DEFAULT_MODEL = "distilbert-base-uncased"
CANDIDATE_LABELS = ["BENIGN", "MALICIOUS"]
LABEL_ALIASES = {"benign", "normal", "legitimate"}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --------------------------------------------------------------------------- #
#                         BASIC  PREP   FUNCTIONS                             #
# --------------------------------------------------------------------------- #
def _detect_label_column(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = f.readline().strip().split(",")
    for cand in ("Label", "label", "Class", "class"):
        if cand in header:
            return cand
    return header[-1]


def _row_to_text(example: Dict[str, str], label_key: str) -> str:
    """Convert CSV row → the semicolon-separated string used in your script."""
    parts = [
        f"\"{k.strip()}\":{v}"
        for k, v in example.items()
        if k != label_key
    ]
    return "; ".join(parts)


def _label_to_id(label: str) -> int:
    return 0 if label.strip().lower() in LABEL_ALIASES else 1


# --------------------------------------------------------------------------- #
#                         DATASET  PREPARATION                                #
# --------------------------------------------------------------------------- #
def _build_hf_dataset(csv_path: str, label_key: str, tokenizer, max_length: int):
    ds = load_dataset(
        "csv",
        data_files={"train": csv_path},
        split="train",
    )

    def preprocess(batch):
        texts = [_row_to_text(r, label_key) for r in batch]
        enc = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        labels = [_label_to_id(r[label_key]) for r in batch]
        enc["labels"] = labels
        return enc

    ds = ds.map(
        preprocess,
        batched=True,
        batch_size=1024,
        remove_columns=ds.column_names,
    )
    return ds.train_test_split(test_size=0.2, seed=RANDOM_SEED)


# --------------------------------------------------------------------------- #
#                           METRIC  UTILITIES                                 #
# --------------------------------------------------------------------------- #
_accuracy = evaluate.load("accuracy")
_precision = evaluate.load("precision")
_recall = evaluate.load("recall")
_f1 = evaluate.load("f1")


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": _accuracy.compute(predictions=preds, references=labels)[
            "accuracy"
        ],
        "precision": _precision.compute(predictions=preds, references=labels)[
            "precision"
        ],
        "recall": _recall.compute(predictions=preds, references=labels)["recall"],
        "f1": _f1.compute(predictions=preds, references=labels)["f1"],
    }


# --------------------------------------------------------------------------- #
#                        PUBLIC  TRAIN / INFER                                #
# --------------------------------------------------------------------------- #
def fine_tune_llm(
    dataset_path: str,
    model_name: str,
    output_root: str,
    update_progress_cb=None,
    num_epochs: int = 3,
    max_length: int = 256,
) -> Tuple[str, Dict[str, float], List[List[int]]]:
    t0 = time.time()

    if update_progress_cb:
        update_progress_cb(5)

    label_key = _detect_label_column(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = _build_hf_dataset(dataset_path, label_key, tokenizer, max_length)

    if update_progress_cb:
        update_progress_cb(20)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CANDIDATE_LABELS)
    )

    # number of updates used to scale progress‐bar
    total_steps = math.ceil(len(data["train"]) / 8) * num_epochs
    last_pct = 20

    def _callback(step: int):
        nonlocal last_pct
        pct = 20 + int(step / total_steps * 60)  # 20-80 %
        if pct > last_pct and update_progress_cb:
            update_progress_cb(pct)
            last_pct = pct

    class ProgressCallback(Trainer):
        def on_step_end(self, args, state, control, **kwargs):
            _callback(state.global_step)

    args = TrainingArguments(
        output_dir="./tmp_trainer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        learning_rate=3e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        seed=RANDOM_SEED,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[ProgressCallback],
    )

    trainer.train()

    if update_progress_cb:
        update_progress_cb(85)

    metrics = trainer.evaluate()
    cm = [[0, 0], [0, 0]]  # confusion matrix isn’t returned directly

    # save final artefacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(output_root).joinpath("llm_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_dir.joinpath(f"{Path(dataset_path).stem}__{ts}")
    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    with open(ckpt_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    if update_progress_cb:
        update_progress_cb(100)

    print(
        f"[LLM-TRAIN] {dataset_path} → {ckpt_dir}  "
        f"(took {time.time()-t0:.1f}s, acc={metrics['eval_accuracy']:.3f})"
    )
    return str(ckpt_dir), metrics, cm


def infer_llm(
    ckpt_dir: str,
    dataset_path: str,
    output_root: str,
    update_progress_cb=None,
    max_length: int = 256,
):
    label_key = _detect_label_column(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir, num_labels=len(CANDIDATE_LABELS)
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(dataset_path, low_memory=False)
    has_labels = label_key in df.columns

    if update_progress_cb:
        update_progress_cb(15)

    texts = [_row_to_text(r, label_key) for r in df.to_dict(orient="records")]
    preds = []

    batch_size = 16
    total_batches = math.ceil(len(texts) / batch_size)
    last_pct = 15

    for i in range(total_batches):
        batch_txt = texts[i * batch_size : (i + 1) * batch_size]
        enc = tokenizer(
            batch_txt,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

        # progress
        pct = 15 + int((i + 1) / total_batches * 60)
        if pct > last_pct and update_progress_cb:
            update_progress_cb(pct)
            last_pct = pct

    preds_str = [CANDIDATE_LABELS[p] for p in preds]

    if update_progress_cb:
        update_progress_cb(80)

    metrics, cm = {}, []
    if has_labels:
        y_true = [_label_to_id(l) for l in df[label_key]]
        from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds),
        }
        cm = confusion_matrix(y_true, preds).tolist()

    res_dir = Path(output_root).joinpath("llm_inference_results")
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / f"results__{Path(dataset_path).stem}__{Path(ckpt_dir).stem}.json"
    with open(out_path, "w") as fp:
        json.dump(
            {"metrics": metrics, "confusion_matrix": cm, "predictions": preds_str},
            fp,
            indent=2,
        )

    if update_progress_cb:
        update_progress_cb(100)

    return metrics, cm, str(out_path)
