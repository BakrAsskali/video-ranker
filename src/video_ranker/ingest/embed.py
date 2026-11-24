import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from video_ranker.config import get_settings


@lru_cache
def load_model():
    import open_clip

    s = get_settings()
    if s.hf_token:
        os.environ.setdefault("HF_TOKEN", s.hf_token)
    model, preprocess = open_clip.create_model_from_pretrained(s.clip_model)
    tokenizer = open_clip.get_tokenizer(s.clip_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), preprocess, tokenizer, device


def _normalize(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def embed_images(paths: list[Path], batch_size: int = 32) -> np.ndarray:
    model, preprocess, _, device = load_model()
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        batch = torch.stack([preprocess(im) for im in imgs])
        with torch.autocast(device, enabled=device == "cuda"):
            out.append(_normalize(model.encode_image(batch.to(device))))
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(texts: list[str]) -> np.ndarray:
    model, _, tokenizer, device = load_model()
    with torch.autocast(device, enabled=device == "cuda"):
        return _normalize(model.encode_text(tokenizer(texts).to(device)))


def chunk_words(text: str, size: int = 40, overlap: int = 10) -> list[str]:
    # siglip only sees 64 tokens, so long transcripts get split into overlapping windows
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def embed_document(text: str) -> np.ndarray | None:
    chunks = chunk_words(text)
    if not chunks:
        return None
    v = embed_texts(chunks).mean(axis=0)
    return v / np.linalg.norm(v)
