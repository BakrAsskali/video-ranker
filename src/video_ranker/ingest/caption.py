import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from video_ranker.config import get_settings

PROMPT = "Describe this video frame in one or two sentences."


@lru_cache
def load_vlm():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    s = get_settings()
    if s.hf_token:
        os.environ.setdefault("HF_TOKEN", s.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(s.caption_model)
    model = AutoModelForImageTextToText.from_pretrained(s.caption_model, dtype=dtype)
    return model.to(device).eval(), processor, device


def pick_frames(n_frames: int, k: int = 4) -> list[int]:
    # evenly spread indices, always keep first and last
    if n_frames <= k:
        return list(range(n_frames))
    return sorted({round(i * (n_frames - 1) / (k - 1)) for i in range(k)})


@torch.no_grad()
def caption_images(paths: list[Path], max_new_tokens: int = 60) -> list[str]:
    model, processor, device = load_vlm()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    processor.tokenizer.padding_side = "left"

    imgs = [[Image.open(p).convert("RGB")] for p in paths]
    inputs = processor(text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True)
    inputs = inputs.to(device, dtype=model.dtype)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1] :]
    return [clean(t) for t in processor.batch_decode(new, skip_special_tokens=True)]


def clean(text: str) -> str:
    # smolvlm starts almost every caption with this
    text = text.strip().removeprefix("In this image we can see").strip()
    return text[:1].upper() + text[1:]
