"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from collections import Counter
import math
import os
from types import SimpleNamespace
import inspect
from contextlib import nullcontext

from model import Transformer, make_src_mask, make_tgt_mask

wandb = None
from lr_scheduler import NoamScheduler


def safe_wandb_log(values: dict) -> None:
    if wandb is not None and wandb.run is not None:
        try:
            wandb.log(values)
        except Exception as exc:
            print(f"Warning: W&B log failed ({exc}). Continuing.")


def init_wandb(project: str, config: dict):
    global wandb
    if wandb is None:
        return None
    try:
        return wandb.init(project=project, config=config)
    except Exception as exc:
        print(f"Warning: W&B init failed ({exc}). Continuing with W&B disabled.")
        wandb = None
        return None


def build_multi30k_dataset(dataset_cls, split: str, **kwargs):
    signature = inspect.signature(dataset_cls.__init__)
    accepted = set(signature.parameters)
    filtered_kwargs = {
        key: value for key, value in kwargs.items()
        if key in accepted
    }
    return dataset_cls(split=split, **filtered_kwargs)


def dataloader_source(dataset):
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        return dataset
    if hasattr(dataset, "processed_data"):
        return dataset.processed_data
    return dataset


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        if logits.numel() == 0:
            return logits.sum()

        confidence = 1.0 - self.smoothing

        true_dist = torch.zeros_like(logits)
        true_dist.fill_(
            self.smoothing / (self.vocab_size - 2)
        )

        true_dist.scatter_(
            1,
            target.unsqueeze(1),
            confidence
        )

        true_dist[:, self.pad_idx] = 0

        mask = (target == self.pad_idx)
        true_dist[mask] = 0

        token_loss = torch.sum(
            -true_dist * torch.log_softmax(logits, dim=1),
            dim=1
        )

        non_pad = target != self.pad_idx
        if not non_pad.any():
            return token_loss.sum() * 0.0
        return token_loss[non_pad].mean()


def corpus_bleu_score(predictions, references, max_n=4) -> float:
    clipped_matches = [0] * max_n
    total_counts = [0] * max_n
    pred_len = 0
    ref_len = 0

    for pred_tokens, ref_tokens in zip(predictions, references):
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = Counter(
                tuple(pred_tokens[i:i + n])
                for i in range(max(0, len(pred_tokens) - n + 1))
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i + n])
                for i in range(max(0, len(ref_tokens) - n + 1))
            )

            clipped_matches[n - 1] += sum(
                min(count, ref_ngrams[ngram])
                for ngram, count in pred_ngrams.items()
            )
            total_counts[n - 1] += sum(pred_ngrams.values())

    if pred_len == 0:
        return 0.0

    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(1.0)
        elif clipped_matches[i] == 0:
            precisions.append(1e-9)
        else:
            precisions.append(clipped_matches[i] / total_counts[i])

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    bleu = brevity_penalty * math.exp(
        sum(math.log(p) for p in precisions) / max_n
    )
    return bleu * 100


def vocab_lookup_token(vocab, idx: int) -> str:
    if idx < 0:
        return "<unk>"
    if hasattr(vocab, "itos"):
        return vocab.itos[idx] if idx < len(vocab.itos) else "<unk>"
    if hasattr(vocab, "tgt_itos"):
        return vocab.tgt_itos[idx] if idx < len(vocab.tgt_itos) else "<unk>"
    if isinstance(vocab, (list, tuple)):
        return vocab[idx] if idx < len(vocab) else "<unk>"
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    raise TypeError("tgt_vocab must provide itos, tgt_itos, lookup_token, or be a list")


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    grad_context = nullcontext() if is_train else torch.no_grad()

    for batch in data_iter:
        src, tgt = batch

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        with grad_context:
            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask
            )

            logits = logits.contiguous().view(
                -1,
                logits.size(-1)
            )

            tgt_output = tgt_output.contiguous().view(-1)

            non_pad = tgt_output != 1
            if non_pad.any():
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions[non_pad] == tgt_output[non_pad]).sum().item()
                total_tokens += non_pad.sum().item()

            step_for_logging = getattr(run_epoch, "global_step", 0)
            should_log_batch = (not is_train) or step_for_logging % 50 == 0
            if non_pad.any() and should_log_batch:
                probs = torch.softmax(logits[non_pad], dim=1)
                correct_token_probs = probs.gather(
                    1,
                    tgt_output[non_pad].unsqueeze(1)
                ).squeeze(1)
                confidence = correct_token_probs.mean().item()
                safe_wandb_log({
                    "prediction_confidence": confidence
                })

            loss = loss_fn(
                logits,
                tgt_output
            )

        if is_train:
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss encountered.")

            optimizer.zero_grad()
            loss.backward()

            step = getattr(run_epoch, "global_step", 0) + 1
            run_epoch.global_step = step

            if step <= 1000:
                q_norms = []
                k_norms = []
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if ".W_q." in name:
                        q_norms.append(param.grad.norm().item())
                    elif ".W_k." in name:
                        k_norms.append(param.grad.norm().item())
                safe_wandb_log({
                    "grad_norm_q": sum(q_norms) / max(1, len(q_norms)),
                    "grad_norm_k": sum(k_norms) / max(1, len(k_norms)),
                    "train_step": step,
                })

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient norm encountered.")
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_iter)
    token_accuracy = total_correct / max(1, total_tokens)
    safe_wandb_log({
        "train_token_accuracy" if is_train else "val_token_accuracy": token_accuracy,
        "epoch": epoch_num,
    })
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        ys = torch.ones(
            1,
            1
        ).fill_(start_symbol).type_as(src).to(device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)

            out = model.decode(
                memory,
                src_mask,
                ys,
                tgt_mask
            )

            next_word = torch.argmax(
                out[:, -1],
                dim=1
            ).item()

            ys = torch.cat(
                [
                    ys,
                    torch.ones(1, 1)
                    .type_as(src)
                    .fill_(next_word)
                    .to(device)
                ],
                dim=1
            )

            if next_word == end_symbol:
                break

        return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    model.eval()

    predictions = []
    references = []

    # special token ids
    start_symbol = 2   # <sos>
    end_symbol = 3     # <eos>
    pad_idx = 1        # <pad>

    with torch.no_grad():
        for batch in test_dataloader:
            src_batch, tgt_batch = batch

            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                # one sentence at a time
                src = src_batch[i].unsqueeze(0)
                tgt = tgt_batch[i].tolist()

                src_mask = make_src_mask(src)

                # generate prediction using greedy decoding
                sentence_max_len = min(max_len, int((src != pad_idx).sum().item()) + 50)

                prediction = greedy_decode(
                    model=model,
                    src=src,
                    src_mask=src_mask,
                    max_len=sentence_max_len,
                    start_symbol=start_symbol,
                    end_symbol=end_symbol,
                    device=device,
                )

                pred_tokens = prediction.squeeze(0).tolist()

                # remove special tokens from prediction
                pred_tokens = [
                    int(token) for token in pred_tokens
                    if token not in [start_symbol, end_symbol, pad_idx]
                ]

                # remove special tokens from target
                tgt_tokens = [
                    int(token) for token in tgt
                    if token not in [start_symbol, end_symbol, pad_idx]
                ]

                # convert token ids → words
                pred_words = [
                    vocab_lookup_token(tgt_vocab, idx)
                    for idx in pred_tokens
                ]
                tgt_words = [
                    vocab_lookup_token(tgt_vocab, idx)
                    for idx in tgt_tokens
                ]

                # BLEU format:
                # predictions = ["this is good"]
                # references = [["this is good"]]
                predictions.append(pred_words)
                references.append(tgt_words)

    bleu_score = corpus_bleu_score(predictions, references)

    print(f"Test BLEU Score: {bleu_score:.2f}")

    return bleu_score

# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def log_encoder_attention_heatmaps(
    model: Transformer,
    src_sentence,
    src_itos,
    device: str = "cpu",
    prefix: str = "encoder_attention",
) -> None:
    if wandb is None or wandb.run is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: attention heatmap logging skipped ({exc}).")
        return

    if isinstance(src_sentence, tuple):
        src_sentence = src_sentence[0]

    src_ids = [
        int(token) for token in src_sentence
        if int(token) != 1
    ]
    if not src_ids:
        return

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)
    src_mask = make_src_mask(src)

    model.eval()
    with torch.no_grad():
        model.encode(src, src_mask)

    attn = getattr(model.encoder.layers[-1].self_attn, "attention_weights", None)
    if attn is None:
        print("Warning: attention weights were not available for heatmap logging.")
        return

    attn = attn.detach().cpu()[0]
    tokens = [
        vocab_lookup_token(src_itos, token_id)
        for token_id in src_ids
    ]

    logs = {}
    for head_idx in range(attn.size(0)):
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(attn[head_idx].numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"Last encoder layer head {head_idx}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        logs[f"{prefix}/head_{head_idx}"] = wandb.Image(fig)
        plt.close(fig)

    safe_wandb_log(logs)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": {
                "src_vocab_size": model.src_vocab_size,
                "tgt_vocab_size": model.tgt_vocab_size,
                "d_model": model.d_model,
                "N": model.N,
                "num_heads": model.num_heads,
                "d_ff": model.d_ff,
                "dropout": model.dropout_value,
                "use_learned_positional": model.use_learned_positional,
                "use_scaling": model.use_scaling,
                "tie_embeddings": model.tie_embeddings,
                "max_decode_len": getattr(model, "max_decode_len", 100),
            },
            "src_vocab": getattr(model, "src_vocab", None),
            "tgt_vocab": getattr(model, "tgt_vocab", None),
            "src_itos": getattr(model, "src_itos", None),
            "tgt_itos": getattr(model, "tgt_itos", None),
        },
        path
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    from dataset import Multi30kDataset, collate_fn
    import wandb as wandb_module

    global wandb
    wandb = wandb_module

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B config
    default_config = {
        "batch_size": 32,
        "epochs": 30,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 8000,
        "learning_rate": 0.2,
        "use_noam_scheduler": True, # False
        "fixed_learning_rate": 1e-4,
        "label_smoothing": 0.1,
        "use_scaling": True,
        "use_learned_positional": False,
        "tie_embeddings": False,
        "min_freq": 2,
        "max_vocab_size": None,
        "num_workers": 0,
        "checkpoint_path": "best_checkpoint.pt",
        "max_decode_len": 80,
        "log_attention_heatmaps": True,
        "early_stop_patience": 3,
        "divergence_factor": 1.35,
    }

    run = init_wandb(
        project="da6401_Assigment_03_Weight_&_Bias",
        config=default_config
    )

    config = wandb.config if run is not None else SimpleNamespace(**default_config)

    print(
        "Training config: "
        f"lr={config.learning_rate}, warmup={config.warmup_steps}, "
        f"epochs={config.epochs}, batch_size={config.batch_size}, "
        f"tie_embeddings={config.tie_embeddings}, "
        f"divergence_factor={config.divergence_factor}"
    )

    # dataset
    train_data = build_multi30k_dataset(
        Multi30kDataset,
        split="train",
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    val_data = build_multi30k_dataset(
        Multi30kDataset,
        split="validation",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        src_itos=train_data.src_itos,
        tgt_itos=train_data.tgt_itos
    )

    test_data = build_multi30k_dataset(
        Multi30kDataset,
        split="test",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        src_itos=train_data.src_itos,
        tgt_itos=train_data.tgt_itos
    )

    train_loader = DataLoader(
        dataloader_source(train_data),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        dataloader_source(val_data),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda")
    )

    test_loader = DataLoader(
        dataloader_source(test_data),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda")
    )
    
    # model
    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=config.d_model,
        N=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        use_learned_positional=config.use_learned_positional,
        use_scaling=config.use_scaling,
        tie_embeddings=config.tie_embeddings,
        load_pretrained=False,
        max_decode_len=config.max_decode_len,
    ).to(device)
    model.src_vocab = train_data.src_vocab
    model.tgt_vocab = train_data.tgt_vocab
    model.src_itos = train_data.src_itos
    model.tgt_itos = train_data.tgt_itos

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate if config.use_noam_scheduler else config.fixed_learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # scheduler
    if config.use_noam_scheduler:
        scheduler = NoamScheduler(
            optimizer=optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps
        )
    else:
        scheduler = None
    
    # loss function
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab),
        pad_idx=1,
        smoothing=config.label_smoothing
    )

    # training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs} started")

        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device
        )

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        ) 

        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            print("Non-finite loss detected; stopping and keeping best checkpoint.")
            break

        safe_wandb_log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # save best model only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=config.checkpoint_path
            )

            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if val_loss > best_val_loss * config.divergence_factor:
            print(
                "Validation loss diverged; stopping early and keeping "
                "the best checkpoint."
            )
            break

        if epochs_without_improvement >= config.early_stop_patience:
            print("No validation improvement; stopping early.")
            break

    if os.path.exists(config.checkpoint_path):
        load_checkpoint(config.checkpoint_path, model, device=device)
        model.to(device)
    else:
        print("Warning: best checkpoint was not found; evaluating current model.")

    if getattr(config, "log_attention_heatmaps", True):
        val_source = dataloader_source(val_data)
        log_encoder_attention_heatmaps(
            model,
            val_source[0],
            train_data.src_itos,
            device=device,
        )

    # final BLEU
    bleu = evaluate_bleu(
          model,
          test_loader,
          train_data.tgt_itos,
          device=device,
          max_len=config.max_decode_len)

    safe_wandb_log({
        "test_bleu": bleu
    })

    print(f"Final BLEU Score: {bleu:.2f}")

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
