"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
UNK_IDX = 0


# checkpoint saved by save_checkpoint in train.py after this update.
DEFAULT_PRETRAINED_FILE_ID = ""
DEFAULT_PRETRAINED_PATH = ""


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    if use_scaling:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    else:
        scores = torch.matmul(Q, K.transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    subsequent_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device),
        diagonal=1
    ).bool()

    subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)

    return pad_mask | subsequent_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.use_scaling = use_scaling
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        _, attn = scaled_dot_product_attention(Q, K, V, mask, self.use_scaling)
        attn = self.dropout(attn)
        x = torch.matmul(attn, V)
        self.attention_weights = attn

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)

        return self.W_o(x)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(
            self.dropout(
                F.relu(self.linear1(x))
            )
        )


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        # TODO:instantiate:
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # self attention
        attn_output = self.self_attn(x, x, x, src_mask)

        # add + norm
        x = self.norm1(x + self.dropout(attn_output))

        # feed forward
        ff_output = self.feed_forward(x)

        # add + norm
        x = self.norm2(x + self.dropout(ff_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        # TODO: instantiate:
        # masked self attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)

        # encoder-decoder attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)

        # feed forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # masked self attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # cross attention
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # feed forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        use_learned_positional = False,
        use_scaling: bool = True,
        tie_embeddings: bool = False,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: Optional[str] = None,
        checkpoint_file_id: Optional[str] = None,
        load_pretrained: Optional[bool] = None,
        max_decode_len: int = 100,
    ) -> None:
        super().__init__()
        self.max_decode_len = max_decode_len
        self.src_vocab: Dict[str, int] = {}
        self.tgt_vocab: Dict[str, int] = {}
        self.src_itos: List[str] = []
        self.tgt_itos: List[str] = []
        self.spacy_de = None
        self.spacy_en = None

        artifact = None
        if load_pretrained is None:
            load_pretrained = src_vocab_size is None or tgt_vocab_size is None

        if load_pretrained:
            artifact = self._load_pretrained_artifact(
                checkpoint_path=checkpoint_path,
                checkpoint_file_id=checkpoint_file_id,
            )
            if artifact is not None:
                model_config = artifact.get("model_config", {})
                src_vocab_size = model_config.get("src_vocab_size", src_vocab_size)
                tgt_vocab_size = model_config.get("tgt_vocab_size", tgt_vocab_size)
                d_model = model_config.get("d_model", d_model)
                N = model_config.get("N", model_config.get("num_layers", N))
                num_heads = model_config.get("num_heads", num_heads)
                d_ff = model_config.get("d_ff", d_ff)
                dropout = model_config.get("dropout", dropout)
                use_learned_positional = model_config.get(
                    "use_learned_positional",
                    use_learned_positional
                )
                use_scaling = model_config.get("use_scaling", use_scaling)
                tie_embeddings = model_config.get("tie_embeddings", tie_embeddings)
                max_decode_len = model_config.get("max_decode_len", max_decode_len)
                self.max_decode_len = max_decode_len
                self._restore_vocab_from_artifact(artifact)

        if not self.src_vocab or not self.tgt_vocab:
            self._build_fallback_vocab(src_vocab_size, tgt_vocab_size)

        src_vocab_size = src_vocab_size or len(self.src_itos)
        tgt_vocab_size = tgt_vocab_size or len(self.tgt_itos)

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError("src_vocab_size and tgt_vocab_size could not be inferred.")

        self._load_tokenizers()

        # TODO: Instantiate 
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_value = dropout
        self.use_learned_positional = use_learned_positional
        self.use_scaling = use_scaling
        self.tie_embeddings = tie_embeddings

        # embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        # positional encoding
        if use_learned_positional:
            self.src_position_embedding = nn.Embedding(5000, d_model)
            self.tgt_position_embedding = nn.Embedding(5000, d_model)
            self.position_dropout = nn.Dropout(dropout)
        else:
            self.positional_encoding = PositionalEncoding(
                d_model,
                dropout
            )

        # encoder
        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            use_scaling
        )

        self.encoder = Encoder(
            encoder_layer,
            N
        )

        # decoder
        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            use_scaling
        )

        self.decoder = Decoder(
            decoder_layer,
            N
        )

        # final output projection
        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self._reset_parameters()
        with torch.no_grad():
            self.src_embedding.weight[PAD_IDX].zero_()
            self.tgt_embedding.weight[PAD_IDX].zero_()
        if tie_embeddings:
            self.fc_out.weight = self.tgt_embedding.weight

        if artifact is not None and "model_state_dict" in artifact:
            self.load_state_dict(artifact["model_state_dict"], strict=True)

    def _load_pretrained_artifact(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_file_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        path = (
            checkpoint_path
            or os.environ.get("TRANSFORMER_ARTIFACT_PATH")
            or DEFAULT_PRETRAINED_PATH
        )
        file_id = (
            checkpoint_file_id
            or os.environ.get("TRANSFORMER_ARTIFACT_FILE_ID")
            or DEFAULT_PRETRAINED_FILE_ID
        )

        if not os.path.exists(path) and file_id:
            try:
                import gdown
            except ImportError as exc:
                raise ImportError(
                    "gdown is required to download pretrained weights. "
                    "Add it to requirements.txt or upload a local artifact."
                ) from exc

            gdown.download(id=file_id, output=path, quiet=False, fuzzy=True)

        if not os.path.exists(path):
            return None

        return torch.load(path, map_location="cpu")

    def _restore_vocab_from_artifact(self, artifact: Dict[str, Any]) -> None:
        self.src_itos = list(artifact.get("src_itos") or [])
        self.tgt_itos = list(artifact.get("tgt_itos") or [])

        self.src_vocab = dict(artifact.get("src_vocab") or {})
        self.tgt_vocab = dict(artifact.get("tgt_vocab") or {})

        if not self.src_vocab and self.src_itos:
            self.src_vocab = {token: idx for idx, token in enumerate(self.src_itos)}
        if not self.tgt_vocab and self.tgt_itos:
            self.tgt_vocab = {token: idx for idx, token in enumerate(self.tgt_itos)}

    def _build_fallback_vocab(
        self,
        src_vocab_size: Optional[int],
        tgt_vocab_size: Optional[int],
    ) -> None:
        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        src_size = max(src_vocab_size or len(specials), len(specials))
        tgt_size = max(tgt_vocab_size or len(specials), len(specials))

        self.src_itos = specials + [
            f"<src_extra_{idx}>"
            for idx in range(src_size - len(specials))
        ]
        self.tgt_itos = specials + [
            f"<tgt_extra_{idx}>"
            for idx in range(tgt_size - len(specials))
        ]
        self.src_vocab = {token: idx for idx, token in enumerate(self.src_itos)}
        self.tgt_vocab = {token: idx for idx, token in enumerate(self.tgt_itos)}

    def _load_tokenizers(self) -> None:
        try:
            import spacy

            try:
                self.spacy_de = spacy.load("de_core_news_sm")
            except OSError:
                self.spacy_de = spacy.blank("de")

            try:
                self.spacy_en = spacy.load("en_core_web_sm")
            except OSError:
                self.spacy_en = spacy.blank("en")
        except Exception:
            self.spacy_de = None
            self.spacy_en = None

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        # x = self.positional_encoding(x)
        if self.use_learned_positional:
            positions = torch.arange(
                0,
                src.size(1),
                device=src.device
            ).unsqueeze(0)

            x = x + self.src_position_embedding(positions)
            x = self.position_dropout(x)
        else:
            x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        if self.use_learned_positional:
            positions = torch.arange(
                0,
                tgt.size(1),
                device=tgt.device
            ).unsqueeze(0)

            x = x + self.tgt_position_embedding(positions)
            x = self.position_dropout(x)
        else:
            x = self.positional_encoding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        return output

    def _tokenize_de(self, sentence: str) -> List[str]:
        if self.spacy_de is not None:
            return [tok.text.lower() for tok in self.spacy_de.tokenizer(sentence)]
        return sentence.lower().strip().split()

    def _ids_to_english(self, token_ids: List[int]) -> str:
        words = []
        for token_id in token_ids:
            if token_id in (SOS_IDX, EOS_IDX, PAD_IDX):
                continue
            if 0 <= token_id < len(self.tgt_itos):
                token = self.tgt_itos[token_id]
            else:
                token = "<unk>"
            if token != "<unk>":
                words.append(token)
        return " ".join(words)

    def infer(self, german_sentence: str) -> str:
        """
        Translate one German sentence into English.

        This is the end-to-end inference hook expected by the autograder:
        tokenize German text, run encoder-decoder autoregressive decoding,
        detokenize target ids, and return a single English sentence.
        """
        if not isinstance(german_sentence, str):
            raise TypeError("german_sentence must be a string")

        device = next(self.parameters()).device
        tokens = self._tokenize_de(german_sentence)
        src_ids = [SOS_IDX]
        src_ids.extend(self.src_vocab.get(token, UNK_IDX) for token in tokens)
        src_ids.append(EOS_IDX)

        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, PAD_IDX)

        was_training = self.training
        self.eval()
        generated = [SOS_IDX]

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            max_len = max(2, self.max_decode_len)

            for _ in range(max_len - 1):
                tgt = torch.tensor(
                    generated,
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)
                tgt_mask = make_tgt_mask(tgt, PAD_IDX)
                logits = self.decode(memory, src_mask, tgt, tgt_mask)
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                generated.append(next_id)
                if next_id == EOS_IDX:
                    break

        if was_training:
            self.train()

        return self._ids_to_english(generated)
