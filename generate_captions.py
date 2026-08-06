"""Generate multi-captions for recap-datacomp using Qwen3-VL-8B-Instruct.

Optimised for throughput: batch generation + flash attention.
"""

import io
import os
import sys
import json
import tarfile
import argparse
import time
from pathlib import Path

import torch

_orig_rf = torch.library.register_fake
def _safe_rf(*a, **kw):
    try:
        return _orig_rf(*a, **kw)
    except RuntimeError:
        return lambda fn: fn
torch.library.register_fake = _safe_rf

import torchvision  # noqa: F401
from PIL import Image

SHORT_PROMPT = "Describe the image briefly in one sentence."
LONG_PROMPT = "Describe the image in detail."
MAX_NEW_SHORT = 48
MAX_NEW_LONG = 128
BATCH_SIZE = 8


def load_model(model_path, gpu_id):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[GPU] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"  # required for batch generation

    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map=f"cuda:{gpu_id}",
            )
            print(f"[GPU] attention: {impl}", flush=True)
            break
        except Exception as e:
            print(f"[GPU] {impl} failed: {e}", flush=True)

    model.eval()
    print(f"[GPU] model ready", flush=True)
    return model, processor


@torch.inference_mode()
def gen_batch(model, processor, images, prompt, device, max_new, temperature=0.8):
    """Generate one caption per image in a single batched forward pass."""
    texts = []
    for img in images:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(t)

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    decoded = processor.batch_decode(out, skip_special_tokens=True)
    captions = []
    for text in decoded:
        parts = text.rsplit("assistant", 1)
        cap = parts[-1].strip() if len(parts) > 1 else text.strip()
        captions.append(cap)
    return captions


def process_tar(model, processor, tar_path, output_dir, device, gpu_id):
    tar_name = Path(tar_path).stem
    out_file = Path(output_dir) / f"{tar_name}.captions.json"

    results = {}
    if out_file.exists():
        results = json.loads(out_file.read_text())
        print(f"[GPU {gpu_id}] {tar_name}: resume, {len(results)} done", flush=True)

    tf = tarfile.open(tar_path)
    members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
    total = len(members)

    t0 = time.time()
    done = 0
    pending_keys, pending_imgs = [], []

    def flush_batch():
        if not pending_imgs:
            return
        try:
            shorts = gen_batch(model, processor, pending_imgs, SHORT_PROMPT, device, MAX_NEW_SHORT)
            longs = gen_batch(model, processor, pending_imgs, LONG_PROMPT, device, MAX_NEW_LONG)
            for k, s, l in zip(pending_keys, shorts, longs):
                results[k] = [s, l]
        except Exception as e:
            print(f"[GPU {gpu_id}] batch error: {e}, falling back to single", flush=True)
            for k, img in zip(pending_keys, pending_imgs):
                try:
                    s = gen_batch(model, processor, [img], SHORT_PROMPT, device, MAX_NEW_SHORT)[0]
                    l = gen_batch(model, processor, [img], LONG_PROMPT, device, MAX_NEW_LONG)[0]
                    results[k] = [s, l]
                except Exception as e2:
                    print(f"[GPU {gpu_id}] {k}: ERROR {e2}", flush=True)
                    results[k] = []
        pending_keys.clear()
        pending_imgs.clear()

    for idx, m in enumerate(members):
        key = m.name.replace(".jpg", "")
        if key in results:
            done += 1
            continue
        try:
            img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            pending_keys.append(key)
            pending_imgs.append(img)
        except Exception as e:
            results[key] = []
            done += 1

        if len(pending_imgs) >= BATCH_SIZE:
            flush_batch()
            done += len(pending_keys)  # won't be exact but close

        if (idx + 1) % 200 == 0 or idx == total - 1:
            flush_batch()
            out_file.write_text(json.dumps(results, ensure_ascii=False))
            done = len(results)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"[GPU {gpu_id}] {tar_name}: {done}/{total} "
                  f"({rate:.1f} img/s, ETA {eta:.0f}s)", flush=True)

    tf.close()
    out_file.write_text(json.dumps(results, ensure_ascii=False))
    print(f"[GPU {gpu_id}] {tar_name}: done ({total})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    os.makedirs(args.output_dir, exist_ok=True)

    tars = sorted(str(p) for p in Path(args.tar_dir).glob("*.tar"))
    my_tars = [t for i, t in enumerate(tars) if i % args.num_gpus == args.gpu]
    print(f"[GPU {args.gpu}] {len(my_tars)} tars assigned", flush=True)
    if not my_tars:
        return

    model, processor = load_model(args.model_path, 0)

    for tar_path in my_tars:
        tar_name = Path(tar_path).stem
        out_file = Path(args.output_dir) / f"{tar_name}.captions.json"
        if out_file.exists():
            existing = json.loads(out_file.read_text())
            tf = tarfile.open(tar_path)
            total = sum(1 for m in tf.getmembers() if m.name.endswith(".jpg"))
            tf.close()
            if len(existing) >= total:
                print(f"[GPU {args.gpu}] {tar_name}: complete, skip", flush=True)
                continue
        process_tar(model, processor, tar_path, args.output_dir, f"cuda:0", args.gpu)

    print(f"[GPU {args.gpu}] all done", flush=True)


if __name__ == "__main__":
    main()
