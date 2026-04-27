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

from model import Transformer, make_src_mask, make_tgt_mask

import wandb
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler


def safe_wandb_log(values: dict) -> None:
    if wandb.run is not None:
        wandb.log(values)


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

    smooth = 1e-9
    precisions = [
        (clipped_matches[i] + smooth) / (total_counts[i] + smooth)
        for i in range(max_n)
    ]

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    bleu = brevity_penalty * math.exp(
        sum(math.log(p) for p in precisions) / max_n
    )
    return bleu * 100


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

    for batch in data_iter:
        src, tgt = batch

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        with torch.set_grad_enabled(is_train):
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

            probs = torch.softmax(logits, dim=1)
            non_pad = tgt_output != 1
            correct_token_probs = probs.gather(1, tgt_output.unsqueeze(1)).squeeze(1)
            if non_pad.any():
                confidence = correct_token_probs[non_pad].mean().item()
                safe_wandb_log({
                    "prediction_confidence": confidence
                })

            loss = loss_fn(
                logits,
                tgt_output
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_iter)
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
                prediction = greedy_decode(
                    model=model,
                    src=src,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=start_symbol,
                    end_symbol=end_symbol,
                    device=device,
                )

                pred_tokens = prediction.squeeze(0).tolist()

                # remove special tokens from prediction
                pred_tokens = [
                    token for token in pred_tokens
                    if token not in [start_symbol, end_symbol, pad_idx]
                ]

                # remove special tokens from target
                tgt_tokens = [
                    token for token in tgt
                    if token not in [start_symbol, end_symbol, pad_idx]
                ]

                # convert token ids → words
                if isinstance(tgt_vocab, list):
                    pred_words = [
                        tgt_vocab[idx]
                        for idx in pred_tokens
                        if idx < len(tgt_vocab.itos)
                    ]

                    tgt_words = [
                        tgt_vocab.itos[idx]
                        for idx in tgt_tokens
                        if idx < len(tgt_vocab.itos)
                    ]

                else:
                    pred_words = [
                        tgt_vocab.lookup_token(idx)
                        for idx in pred_tokens
                    ]

                    tgt_words = [
                        tgt_vocab.lookup_token(idx)
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
            },
        },
        path
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
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
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
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
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B config
    wandb.init(
        project="da6401_Assigment_03_Weight_&_Bias",
        config={
            "batch_size": 32,
            "epochs": 10,
            "d_model": 512,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,
            "warmup_steps": 4000,
            "learning_rate": 1.0,
            "use_noam_scheduler": True, # False
            "fixed_learning_rate": 1e-4,
            "label_smoothing": 0.1,
            "use_scaling": True,
            "use_learned_positional": False,
            "min_freq": 2,
            "max_vocab_size": None,
        }
    )

    config = wandb.config


    # dataset
    train_data = Multi30kDataset(
        split="train",
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    val_data = Multi30kDataset(
        split="validation",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        src_itos=train_data.src_itos,
        tgt_itos=train_data.tgt_itos
    )

    test_data = Multi30kDataset(
        split="test",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        src_itos=train_data.src_itos,
        tgt_itos=train_data.tgt_itos
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
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
    ).to(device)

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

        safe_wandb_log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # save best model only
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path="best_checkpoint.pt"
            )

            print("Best model saved.")

    load_checkpoint("best_checkpoint.pt", model)

    # final BLEU
    bleu = evaluate_bleu(
          model,
          test_loader,
          train_data.tgt_itos,
          device=device)

    safe_wandb_log({
        "test_bleu": bleu
    })

    print(f"Final BLEU Score: {bleu:.2f}")

    wandb.finish()        


if __name__ == "__main__":
    run_training_experiment()
