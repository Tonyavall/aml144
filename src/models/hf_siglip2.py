"""hugging face siglip-2 feature extraction for the frozen blend branch.

loads a siglip2 checkpoint (vision + text towers), extracts l2-normalized
shared-space image embeddings and class-prompt text embeddings, and caches
everything per backbone in a single npz so reruns skip the model entirely.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def _l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _prompt_hash(prompts):
    return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()


def _load_model(model_id, device):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device).eval()
    return model, processor


@torch.no_grad()
def _image_embeds(model, processor, paths, batch_size, device):
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt")
        pv = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
        feats = model.get_image_features(pixel_values=pv)
        # transformers 5.x returns a model output object here; the shared-space embedding is its pooler output
        if not torch.is_tensor(feats):
            feats = feats.pooler_output
        out.append(feats.float().cpu().numpy())
    return _l2(np.concatenate(out).astype(np.float32))


@torch.no_grad()
def _text_embeds(model, processor, prompts, device):
    inputs = processor(text=prompts, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)
    if not torch.is_tensor(feats):
        feats = feats.pooler_output
    return _l2(feats.float().cpu().numpy().astype(np.float32))


def zero_shot_probs(img_embeds, txt_embeds, logit_scale_exp, logit_bias):
    # siglip pairwise logits are exp(scale) * cosine + bias; softmax over the
    # class axis turns them into a k-way classifier (the bias shifts every
    # logit equally so it cannot change the argmax, kept for fidelity)
    logits = logit_scale_exp * img_embeds @ txt_embeds.T + logit_bias
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def backbone_pack(
    model_id, tag, train_paths, test_paths, prompts, batch_size, device, cache_dir
):
    """returns train/test/text embeds plus logit scale and bias for a backbone.

    cached at cache_dir/hf_{tag}.npz keyed by a hash of the prompts, so a
    class-name edit invalidates the text embeds and triggers a recompute.
    """
    cache = Path(cache_dir) / f"hf_{tag}.npz"
    h = _prompt_hash(prompts)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["prompt_hash"]) == h:
            return {k: z[k] for k in ("train", "test", "text", "scale_exp", "bias")}
    model, processor = _load_model(model_id, device)
    pack = {
        "train": _image_embeds(model, processor, train_paths, batch_size, device),
        "test": _image_embeds(model, processor, test_paths, batch_size, device),
        "text": _text_embeds(model, processor, prompts, device),
        "scale_exp": np.float32(model.logit_scale.exp().item()),
        "bias": np.float32(model.logit_bias.item()),
    }
    del model
    torch.cuda.empty_cache()
    np.savez(cache, prompt_hash=h, **pack)
    return pack
