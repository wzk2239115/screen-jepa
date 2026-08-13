"""Generate labeled bounding boxes for recap-datacomp using Qwen3-VL-8B.

Produces object-level annotations (label + box coordinates) per image,
enabling coordinate-aware JEPA training (DeepSeek visual primitives).

Output: per-tar JSON
  { "000000000041": [{"label":"person","box":[x1,y1,x2,y2]}, ...] }
  Coordinates are 0-100 (percentage of image dimension).
"""

import io
import os
import re
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

PROMPT = (
    "Identify the main objects and visual elements in this image. "
    "For each, provide a short label and bounding box coordinates. "
    "Output ONLY a JSON array (max 8 items), format: "
    '[{"label":"name","box":[x1,y1,x2,y2]}] '
    "where all coordinates are percentages 0-100. "
    "Include people, text regions, prominent objects, and distinctive features."
)
MAX_NEW = 256
BATCH_SIZE = 8


def load_model(model_path):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[GPU] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map="cuda:0",
            )
            print(f"[GPU] attention: {impl}", flush=True)
            break
        except Exception as e:
            print(f"[GPU] {impl} failed: {e}", flush=True)

    model.eval()
    print(f"[GPU] model ready", flush=True)
    return model, processor


def parse_boxes(text):
    """Extract bounding boxes from model output. Robust to formatting."""
    # Try JSON parse first
    try:
        # Find JSON array in text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            items = json.loads(match.group())
            results = []
            for item in items:
                if isinstance(item, dict) and "box" in item:
                    box = item["box"]
                    label = item.get("label", "object")
                    if len(box) == 4 and all(0 <= float(v) <= 100 for v in box):
                        results.append({"label": str(label)[:50], "box": [float(v) for v in box]})
            if results:
                return results
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex for coordinate pairs
    nums = re.findall(r'\d+\.?\d*', text)
    results = []
    for i in range(0, len(nums) - 3, 4):
        box = [float(nums[j]) for j in range(i, i + 4)]
        if all(0 <= v <= 100 for v in box):
            results.append({"label": "object", "box": box})
    return results[:8]


@torch.inference_mode()
def gen_batch(model, processor, images, device):
    """Generate bounding box annotations for a batch of images."""
    texts = []
    for img in images:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]}]
        t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(t)

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=False,  # greedy for structured output
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    decoded = processor.batch_decode(out, skip_special_tokens=True)

    all_boxes = []
    for text in decoded:
        parts = text.rsplit("assistant", 1)
        raw = parts[-1].strip() if len(parts) > 1 else text.strip()
        all_boxes.append(parse_boxes(raw))
    return all_boxes


def process_tar(model, processor, tar_path, output_dir, device, gpu_id):
    tar_name = Path(tar_path).stem
    out_file = Path(output_dir) / f"{tar_name}.boxes.json"

    results = {}
    if out_file.exists():
        results = json.loads(out_file.read_text())
        print(f"[GPU {gpu_id}] {tar_name}: resume, {len(results)} done", flush=True)

    tf = tarfile.open(tar_path)
    members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
    total = len(members)

    t0 = time.time()
    done = len(results)
    pending_keys, pending_imgs = [], []

    def flush_batch():
        nonlocal done
        if not pending_imgs:
            return
        try:
            boxes_batch = gen_batch(model, processor, pending_imgs, device)
            for k, boxes in zip(pending_keys, boxes_batch):
                results[k] = boxes
        except Exception as e:
            print(f"[GPU {gpu_id}] batch error: {e}, single fallback", flush=True)
            for k, img in zip(pending_keys, pending_imgs):
                try:
                    boxes = gen_batch(model, processor, [img], device)[0]
                    results[k] = boxes
                except Exception as e2:
                    print(f"[GPU {gpu_id}] {k}: ERROR {e2}", flush=True)
                    results[k] = []
        done = len(results)
        pending_keys.clear()
        pending_imgs.clear()

    for idx, m in enumerate(members):
        key = m.name.replace(".jpg", "")
        if key in results:
            continue
        try:
            img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            pending_keys.append(key)
            pending_imgs.append(img)
        except:
            results[key] = []

        if len(pending_imgs) >= BATCH_SIZE:
            flush_batch()

        if (idx + 1) % 200 == 0 or idx == total - 1:
            flush_batch()
            out_file.write_text(json.dumps(results, ensure_ascii=False))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            avg_boxes = sum(len(v) for v in results.values()) / max(done, 1)
            print(f"[GPU {gpu_id}] {tar_name}: {done}/{total} "
                  f"({rate:.1f} img/s, avg {avg_boxes:.1f} boxes/img, ETA {eta:.0f}s)", flush=True)

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

    model, processor = load_model(args.model_path)

    for tar_path in my_tars:
        tar_name = Path(tar_path).stem
        out_file = Path(args.output_dir) / f"{tar_name}.boxes.json"
        if out_file.exists():
            existing = json.loads(out_file.read_text())
            tf = tarfile.open(tar_path)
            total = sum(1 for m in tf.getmembers() if m.name.endswith(".jpg"))
            tf.close()
            if len(existing) >= total:
                print(f"[GPU {args.gpu}] {tar_name}: complete, skip", flush=True)
                continue
        process_tar(model, processor, tar_path, args.output_dir, "cuda:0", args.gpu)

    print(f"[GPU {args.gpu}] all done", flush=True)


if __name__ == "__main__":
    main()
