"""
nlp.py
------
FinBERT NLP pipeline for earnings sentiment scoring.

Uses ProsusAI/finbert from HuggingFace Transformers.

Key design decisions:
  - Long texts are chunked into <= 512-token segments. Scores are averaged
    across chunks (weighted by token count) to handle full earnings transcripts.
  - Batch processing: multiple texts processed in a single forward pass for
    GPU/CPU efficiency.
  - Output schema: for each text, return positive_prob, negative_prob,
    neutral_prob, and composite_score = positive_prob - negative_prob.
    Composite score in [-1, +1]; higher = more bullish.

FinBERT reference:
  Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained
  Language Models. arXiv:1908.10063.
  Model: https://huggingface.co/ProsusAI/finbert
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Model identifier on HuggingFace Hub
FINBERT_MODEL_ID = "ProsusAI/finbert"

# FinBERT was trained on 512-token sequences; we stay safely below the limit
MAX_TOKENS_PER_CHUNK = 480

# Batch size for transformer forward pass (reduce if OOM on small GPU/CPU)
DEFAULT_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_finbert(model_id: str = FINBERT_MODEL_ID, device: str = "cpu"):
    """
    Load FinBERT tokenizer and model from HuggingFace.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (default: ProsusAI/finbert).
    device : str
        'cpu' or 'cuda'. Defaults to cpu for portability.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
    model     : PreTrainedModel  (set to eval mode)

    Raises
    ------
    ImportError
        If `transformers` or `torch` are not installed.
    """
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
    except ImportError as exc:
        raise ImportError(
            "Required packages not found. Install with:\n"
            "  pip install transformers torch\n"
            f"Original error: {exc}"
        ) from exc

    print(f"[nlp] Loading FinBERT from '{model_id}' ...")
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model     = BertForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)
    print(f"[nlp] FinBERT loaded successfully. Device: {device}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
) -> list[str]:
    """
    Split `text` into chunks of at most `max_tokens` tokens.

    Tokens are decoded back to strings so we can pass them through the
    standard tokenizer/model pipeline without custom collation.

    Parameters
    ----------
    text     : str  Input text (may be arbitrarily long).
    tokenizer: HuggingFace tokenizer.
    max_tokens: int  Maximum tokens per chunk (excl. [CLS] / [SEP]).

    Returns
    -------
    list[str]  List of text chunks, each <= max_tokens tokens.
    """
    # Tokenize without special tokens so we can split cleanly
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) == 0:
        return [text]  # edge case: empty or unparseable text

    chunks = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start : start + max_tokens]
        chunk_text_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Single-text scorer
# ---------------------------------------------------------------------------

def score_text(
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
) -> dict[str, float]:
    """
    Score a single text with FinBERT.

    Long texts are chunked; scores are averaged across chunks weighted
    by the number of tokens in each chunk.

    Parameters
    ----------
    text      : str   Input earnings text.
    tokenizer : HuggingFace tokenizer.
    model     : FinBERT model in eval mode.
    device    : str   Torch device string.
    max_tokens: int   Max tokens per chunk.

    Returns
    -------
    dict with keys:
        positive_prob  : float  [0, 1]
        negative_prob  : float  [0, 1]
        neutral_prob   : float  [0, 1]
        composite_score: float  [-1, +1]  = positive_prob - negative_prob
        n_chunks       : int    Number of chunks processed.
    """
    import torch

    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens)

    chunk_probs   = []  # list of np.array([pos, neg, neu]) per chunk
    chunk_weights = []  # token count per chunk (for weighted average)

    for chunk in chunks:
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens + 2,  # +2 for [CLS] and [SEP]
            padding=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        weight  = encoded["input_ids"].shape[1]  # actual token count incl. special

        with torch.no_grad():
            logits = model(**encoded).logits  # shape: (1, 3)

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        chunk_probs.append(probs)
        chunk_weights.append(weight)

    # Weighted average across chunks
    weights_arr = np.array(chunk_weights, dtype=float)
    weights_arr /= weights_arr.sum()

    avg_probs = np.average(np.stack(chunk_probs), weights=weights_arr, axis=0)

    # FinBERT label order: positive=0, negative=1, neutral=2
    # (verified from ProsusAI/finbert model config: id2label)
    pos, neg, neu = float(avg_probs[0]), float(avg_probs[1]), float(avg_probs[2])

    return {
        "positive_prob":   pos,
        "negative_prob":   neg,
        "neutral_prob":    neu,
        "composite_score": pos - neg,  # range [-1, +1]
        "n_chunks":        len(chunks),
    }


# ---------------------------------------------------------------------------
# Batch scorer (main API)
# ---------------------------------------------------------------------------

def score_texts_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score a list of texts in batches.

    Parameters
    ----------
    texts      : list[str]  Input texts to score.
    tokenizer  : HuggingFace tokenizer.
    model      : FinBERT model.
    device     : str        Torch device.
    batch_size : int        Number of texts per batch (memory vs speed tradeoff).
    max_tokens : int        Chunk size.
    verbose    : bool       Print progress every 50 texts.

    Returns
    -------
    pd.DataFrame with columns:
        positive_prob, negative_prob, neutral_prob, composite_score, n_chunks
    Index matches the order of `texts`.
    """
    results = []

    for i, text in enumerate(texts):
        if verbose and i % 50 == 0:
            print(f"[nlp] Scoring text {i+1}/{len(texts)} ...")
        result = score_text(text, tokenizer, model, device=device, max_tokens=max_tokens)
        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pipeline: score an events DataFrame
# ---------------------------------------------------------------------------

def score_events(
    events: pd.DataFrame,
    tokenizer,
    model,
    device: str = "cpu",
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Add FinBERT sentiment columns to an earnings events DataFrame.

    Parameters
    ----------
    events    : pd.DataFrame  Must contain `text_col`.
    tokenizer : HuggingFace tokenizer.
    model     : FinBERT model.
    device    : str           Torch device.
    text_col  : str           Column name containing earnings text.

    Returns
    -------
    pd.DataFrame  Original events DataFrame with added columns:
        positive_prob, negative_prob, neutral_prob, composite_score, n_chunks.
    """
    texts    = events[text_col].tolist()
    scores   = score_texts_batch(texts, tokenizer, model, device=device)
    scores.index = events.index
    return pd.concat([events, scores], axis=1)


# ---------------------------------------------------------------------------
# Sentiment label helper
# ---------------------------------------------------------------------------

def label_from_composite(composite_score: float, threshold: float = 0.1) -> str:
    """
    Convert composite score to categorical label.

    Parameters
    ----------
    composite_score : float  Output of positive_prob - negative_prob.
    threshold       : float  Magnitude threshold for positive/negative.

    Returns
    -------
    str  'positive', 'negative', or 'neutral'.
    """
    if composite_score > threshold:
        return "positive"
    elif composite_score < -threshold:
        return "negative"
    else:
        return "neutral"
