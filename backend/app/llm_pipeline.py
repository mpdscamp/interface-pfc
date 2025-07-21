import os, csv, glob, math, datetime, random, sys, time
from typing import Tuple, Dict, List, Callable

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# hard‑coded for now
CANDIDATE_LABELS = [
    "Benign", "Botnet", "Brute_Force_Attack",
    "DoS_Attack", "Port_Scan_Infiltration",
    "Web_Attack", "Other"
]


class FineTuner:
    def __init__(
        self,
        csv_path: str,
        model_name: str,
        output_root: str,
        update_progress_cb: Callable[[Dict], None],
        num_epochs: int,
        max_length: int,
    ):
        # settings (keep your old constants here or parameterize later)
        self.csv_path      = csv_path
        self.model_name    = model_name
        self.output_root   = output_root
        self.num_epochs    = num_epochs
        self.max_length    = max_length
        self.batch_size    = 8
        self.learning_rate = 3e-5
        self.weight_decay  = 0.01
        self.warmup_ratio  = 0.05
        self.seed          = 21023 + 21041

        # checkpoint & log dirs
        self.ckpt_dir = os.path.join(output_root, "checkpoints")
        self.log_dir  = os.path.join(self.ckpt_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir,  exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "checkpoint_log.txt")

        # progress callback
        self.update_progress_cb = update_progress_cb

        # control flags
        self._save_request = False
        self._stop_request = False

        # device, tokenizer, model
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_name, num_labels=len(CANDIDATE_LABELS))
            .to(self.device)
        )

        # build CSV offset index
        with open(self.csv_path, "r", newline="") as f:
            header_line = f.readline()
            self.header = next(csv.reader([header_line]))
            self.offsets = []
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)

        self.total_rows        = len(self.offsets)
        self.batches_per_epoch = math.ceil(self.total_rows / self.batch_size)
        self.total_steps       = self.batches_per_epoch * self.num_epochs
        self.warmup_steps      = int(self.total_steps * self.warmup_ratio)

    def train(self) -> Tuple[str, Dict[str, float], List[List[int]]]:
        self.start_time = time.time()
        # prepare optimizer, scheduler, scaler
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = scaler

        history: List[List[int]] = []
        metrics: Dict[str, float] = {}

        for epoch in range(self.num_epochs):
            self.model.train()
            # deterministic shuffle
            random.seed(self.seed + epoch)
            indices = list(range(self.total_rows))
            random.shuffle(indices)

            epoch_loss = 0.0
            batches_done = 0

            # DataLoader
            dataset = IndexedCSV(
                csv_path   = self.csv_path,
                offsets    = self.offsets,
                header     = self.header,
                tokenizer  = self.tokenizer,
                max_length = self.max_length,
                indices    = indices,
            )
            loader = DataLoader(
                dataset,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = 4,
                pin_memory  = True,
            )

            pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(pbar):    
                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(device_type=self.device.type):
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                loss_val    = loss.item()
                epoch_loss += loss_val
                batches_done += 1
                avg_loss    = epoch_loss / batches_done

                # progress callback
                step     = epoch * self.batches_per_epoch + (batch_idx + 1)
                progress = step / self.total_steps * 100
                elapsed  = time.time() - self.start_time
                eta      = (elapsed / step) * (self.total_steps - step) if step > 0 else None

                if self.update_progress_cb:
                    self.update_progress_cb({
                        "progress":     progress,
                        "loss":         avg_loss,
                        "epoch":        epoch+1,
                        "batch":        batch_idx+1,
                        "elapsed":      elapsed,
                        "eta":          eta
                    })

                # history.append([epoch+1, batch_idx+1, avg_loss])

                if self._save_request:
                    self._do_checkpoint(epoch, batch_idx, avg_loss, epoch_loss, batches_done)
                    self._save_request = False

                if self._stop_request:
                    self._do_checkpoint(epoch, batch_idx, avg_loss, epoch_loss, batches_done)
                    return (self._last_ckpt_path, metrics, history)

                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            self._do_checkpoint(epoch, None, avg_loss, epoch_loss, batches_done)

        metrics["final_avg_loss"] = avg_loss
        return (self._last_ckpt_path, metrics, history)

    def request_save(self):
        """Called by save_checkpoint_llm()"""
        self._save_request = True

    def request_stop(self):
        """Called by stop_training_llm()"""
        self._stop_request = True

    def _do_checkpoint(
        self,
        epoch: int,
        batch_idx: int = None,
        loss_val: float = None,
        epoch_loss: float = None,
        batches_done: int = None
    ):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"checkpoint-epoch{epoch+1}"
        if batch_idx is not None:
            fname += f"-batch{batch_idx+1}"
        fname += f"-{now}.pt"
        path = os.path.join(self.ckpt_dir, fname)

        try:
            torch.save({
                "epoch":                 epoch+1,
                "batch_idx":             batch_idx,
                "model_state_dict":      self.model.state_dict(),
                "optimizer_state_dict":  self.model.optimizer.state_dict(),
                "scheduler_state_dict":  self.model.scheduler.state_dict(),
                "scaler_state_dict":     self.model.scaler.state_dict(),
                "epoch_loss":            epoch_loss,
                "batches_done":          batches_done,
                "total_steps":           self.total_steps,
                "warmup_steps":          self.warmup_steps,
            }, path)

            with open(self.log_file, "a") as log:
                line = f"{datetime.datetime.now().isoformat()} | epoch {epoch+1}"
                if batch_idx is not None:
                    line += f" | batch {batch_idx+1}"
                if loss_val is not None:
                    line += f" | loss {loss_val:.4f}"
                log.write(line + "\n")

            print(f"[checkpoint saved] {path}")
            self._last_ckpt_path = path

        except Exception as e:
            print(f"[⚠️ checkpoint failed] {e}", file=sys.stderr)

_active_tuners: dict[str, FineTuner] = {}


def fine_tune_llm(
    job_id: str,
    dataset_path: str,
    model_name: str,
    output_root: str,
    update_progress_cb: Callable[[Dict], None] = None,
    num_epochs: int = 1,
    max_length: int = 256,
) -> Tuple[str, Dict[str, float], List[List[int]]]:
    """
    Trains a HuggingFace model on a CSV
    Returns: (last_checkpoint_path, metrics_dict, history_batches)
    """
    tuner = FineTuner(
        csv_path=dataset_path,
        model_name=model_name,
        output_root=output_root,
        update_progress_cb=update_progress_cb,
        num_epochs=num_epochs,
        max_length=max_length,
    )
    _active_tuners[job_id] = tuner

    try:
        return tuner.train()
    finally:
        _active_tuners.pop(job_id, None)


def save_checkpoint_llm(job_id: str):
    """
    External entry-point: request an immediate checkpoint.
    """
    tuner = _active_tuners.get(job_id)
    if tuner is None:
        raise RuntimeError(f"No active training session for job {job_id}")
    tuner.request_save()


def stop_training_llm(job_id: str):
    """
    External entry-point: request checkpoint + graceful stop.
    """
    tuner = _active_tuners.get(job_id)
    if tuner is None:
        raise RuntimeError(f"No active training session for job {job_id}")
    tuner.request_stop()


class IndexedCSV(Dataset):
    def __init__(self, csv_path, offsets, header, tokenizer, max_length, indices):
        self.csv_path   = csv_path
        self.offsets    = offsets
        self.header     = header
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.indices    = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        off = self.offsets[self.indices[idx]]
        with open(self.csv_path, "r", newline="") as f:
            f.seek(off)
            reader = csv.DictReader(f, fieldnames=self.header)
            row = next(reader)

        # reconstruct the text
        feats = {
            k: v
            for k, v in row.items()
            if k.strip().upper() != "LABEL"
        }
        text = "; ".join(f"\"{k.strip()}\":{v}" for k, v in feats.items())

        # map to binary label (your old logic)
        lab = row[next(h for h in self.header
                       if "label" in h.strip().lower())].strip().upper()
        label = 0 if "BENIGN" in lab else 1

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }
