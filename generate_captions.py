"""Generate multi-captions for recap-datacomp using Qwen3-VL-8B-Instruct.

Produces 8 captions per image (4 short + 4 detailed) for TC-JEPA multi-caption
training. Saves per-tar JSON files for resumability.

Usage (8 GPUs, one process per GPU):
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python generate_captions.py \
            --model_path /home/jovyan/h800fast/wangzekai/Qwen3-VL-8B-Instruct \
            --tar_dir /home/jovyan/h800fast/wangzekai/recap-datacomp-384-1M \
            --output_dir /home/jovyan/h800fast/wangzekai/recap-multicap \
            --gpu $i --num_gpus 8 &
    done
    wait
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

# ---- Fix: torchvision C++ ops don't match NVIDIA custom torch ----
# Patch register_fake to silently skip missing operators instead of crashing
_orig_rf = torch.library.register_fake
def _safe_rf(*a, **kw):
    try:
        return _orig_rf(*a, **kw)
    except RuntimeError:
        return lambda fn: fn  # no-op decorator
torch.library.register_fake = _safe_rf

import torchvision  # now safe to import
from PIL import Image


SHORT_PROMPT = "Describe the image briefly in one sentence."
LONG_PROMPT = "Describe the image in detail."

NUM_SHORT = 1
NUM_LONG = 1
MAX_NEW_SHORT = 48
MAX_NEW_LONG = 128


def load_model(model_path, gpu_id):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[GPU {gpu_id}] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
    )
    model.eval()
    print(f"[GPU {gpu_id}] model ready", flush=True)
    return model, processor


@torch.no_grad()
def gen_one(model, processor, image, prompt, device, max_new=128, temperature=0.8):
    """Generate a single caption."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    # Extract only the generated part (after the input)
    generated = out[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(generated, skip_special_tokens=True).strip()
    return caption


def process_tar(model, processor, tar_path, output_dir, device, gpu_id):
    """Process one tar file: generate 8 captions per image."""
    tar_name = Path(tar_path).stem  # e.g. "data-00000"
    out_file = Path(output_dir) / f"{tar_name}.captions.json"

    # Load existing results for resumability
    results = {}
    if out_file.exists():
        results = json.loads(out_file.read_text())
        print(f"[GPU {gpu_id}] {tar_name}: resuming, {len(results)} already done", flush=True)

    tf = tarfile.open(tar_path)
    members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
    total = len(members)

    t0 = time.time()
    for idx, m in enumerate(members):
        key = m.name.replace(".jpg", "")
        if key in results:
            continue

        try:
            img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")

            captions = []
            # 4 short captions
            for _ in range(NUM_SHORT):
                cap = gen_one(model, processor, img, SHORT_PROMPT, device,
                              max_new=MAX_NEW_SHORT, temperature=0.8)
                captions.append(cap)
            # 4 detailed captions
            for _ in range(NUM_LONG):
                cap = gen_one(model, processor, img, LONG_PROMPT, device,
                              max_new=MAX_NEW_LONG, temperature=0.8)
                captions.append(cap)

            results[key] = captions
        except Exception as e:
            print(f"[GPU {gpu_id}] {tar_name}/{key}: ERROR {e}", flush=True)
            results[key] = []  # empty list for failed images

        # Save every 50 images
        if (idx + 1) % 50 == 0 or idx == total - 1:
            out_file.write_text(json.dumps(results, ensure_ascii=False))
            elapsed = time.time() - t0
            done = idx + 1
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"[GPU {gpu_id}] {tar_name}: {done}/{total} "
                  f"({rate:.1f} img/s, ETA {remaining:.0f}s)", flush=True)

    tf.close()
    out_file.write_text(json.dumps(results, ensure_ascii=False))
    print(f"[GPU {gpu_id}] {tar_name}: done ({total} images)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:0"  # CUDA_VISIBLE_DEVICES handles the mapping

    # Assign tars to this GPU
    tars = sorted(str(p) for p in Path(args.tar_dir).glob("*.tar"))
    my_tars = [t for i, t in enumerate(tars) if i % args.num_gpus == args.gpu]
    print(f"[GPU {args.gpu}] assigned {len(my_tars)}/{len(tars)} tars: "
          f"{Path(my_tars[0]).stem}...{Path(my_tars[-1]).stem}" if my_tars
          else f"[GPU {args.gpu}] no tars assigned", flush=True)

    if not my_tars:
        return

    # Load model
    model, processor = load_model(args.model_path, 0)  # device 0 after CUDA_VISIBLE_DEVICES

    # Process each tar
    for tar_path in my_tars:
        tar_name = Path(tar_path).stem
        out_file = Path(args.output_dir) / f"{tar_name}.captions.json"
        if out_file.exists():
            existing = json.loads(out_file.read_text())
            tf = tarfile.open(tar_path)
            total = sum(1 for m in tf.getmembers() if m.name.endswith(".jpg"))
            tf.close()
            if len(existing) >= total:
                print(f"[GPU {args.gpu}] {tar_name}: already complete, skipping", flush=True)
                continue

        process_tar(model, processor, tar_path, args.output_dir, device, args.gpu)

    print(f"[GPU {args.gpu}] all tars done", flush=True)


if __name__ == "__main__":
    main()
