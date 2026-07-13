"""CLIP zero-shot probe — mirrors probe_zeroshot.py protocol.

Same data, same vocab, same metrics. Only difference:
  - our method: word feature from rendered-pixel feature map
  - CLIP:      word feature from text encoder (tokenize word -> transformer)
"""
import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_clip import CLIPModel, tokenize

STOP = set("""the a an of in on at to for and or but is are was were be been being
with without from by as this that these those it its their his her my your our we
they he she i you very into over under above below up down out off again further
then once here there all any both each few more most other some such no nor not
only own same so than too can will just don should now which who whom whose what
where when why how also has have had do does did doing would could may might must
shall about against between through during before after above below up down out
off over under again further then once here there along across behind beyond during
near outside past round since till until within without two three four five six seven
eight nine ten first second third next last new old big small long great little own
old new look looking looked feel feeling felt get getting got go going went one two
image photo picture close up view background foreground front back side top bottom
left right center middle wall floor ground sky""".split())


def clean_word(w):
    return w.lower().strip(".,!?;:'\"()[]")


def load_samples(tar_dir, n, max_tars=10):
    tars = sorted(Path(tar_dir).glob("*.tar"))[:max_tars]
    out = []
    for tp in tars:
        try:
            tf = tarfile.open(tp)
        except Exception:
            continue
        for m in tf.getmembers():
            if m.name.endswith(".jpg"):
                try:
                    jf = tf.extractfile(m.name.replace(".jpg", ".json"))
                    cap = json.loads(jf.read())["caption"]
                    img_bytes = tf.extractfile(m.name).read()
                    out.append((cap, img_bytes))
                    if len(out) >= n:
                        return out
                except Exception:
                    continue
        tf.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--n_proto", type=int, default=500)
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--min_count", type=int, default=5)
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ck["args"]
    img_size = margs["img_size"]
    print(f"[clip-zs] ckpt={args.ckpt} hidden={margs['hidden']}", flush=True)

    model = CLIPModel(margs["hidden"], img_size, 16,
                      margs.get("text_layers", 8), margs.get("text_heads", 12)).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    samples = load_samples(args.tar_dir, args.n_proto + args.n_test)
    random.seed(0)
    random.shuffle(samples)
    proto_samples = samples[:args.n_proto]
    test_samples = samples[args.n_proto:args.n_proto + args.n_test]
    print(f"[clip-zs] {len(proto_samples)} proto, {len(test_samples)} test", flush=True)

    # === 1. build word vocab from proto samples (same as probe_zeroshot) ===
    word_count = {}
    for cap, _ in proto_samples:
        for w in cap.split():
            w = clean_word(w)
            if w and w not in STOP and len(w) > 2:
                word_count[w] = word_count.get(w, 0) + 1
    word_list = sorted([w for w, c in word_count.items() if c >= args.min_count])
    print(f"[clip-zs] {len(word_list)} words (>= {args.min_count} occurrences)", flush=True)
    print(f"[clip-zs] vocab sample: {word_list[:30]}", flush=True)

    # === 2. word features via text encoder (tokenize each word) ===
    with torch.no_grad():
        tok_list = []
        for w in word_list:
            tok_list.append(tokenize(f"a photo of {w}"))  # CLIP-style prompt
        word_tokens = torch.stack(tok_list).to(device)
        # batch encode to avoid OOM
        word_feats = []
        for i in range(0, len(word_tokens), 64):
            word_feats.append(model.encode_text(word_tokens[i:i+64]))
        word_feats = torch.cat(word_feats, dim=0)  # (num_words, D)

    inter = (word_feats @ word_feats.T)
    idx = torch.triu_indices(len(word_list), len(word_list), offset=1)
    inter_vals = inter[idx[0], idx[1]]
    print(f"[clip-zs] word inter-cos: mean={inter_vals.mean():.4f} std={inter_vals.std():.4f}", flush=True)

    # === 3. zero-shot image->word retrieval ===
    print(f"\n=== zero-shot image->word retrieval ===", flush=True)
    top1, top5, top10, total = 0, 0, 0, 0
    mrr_sum = 0.0
    for cap, img_bytes in test_samples:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((img_size, img_size))
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            img_f = model.encode_image(t)

        sims = word_feats @ img_f.T  # (num_words, 1)
        sims = sims.squeeze(1)
        order = sims.argsort(descending=True)
        ranked = [word_list[i] for i in order]

        gt_words = set(clean_word(w) for w in cap.split()) - STOP
        gt_words = {w for w in gt_words if w in set(word_list)}
        if not gt_words:
            continue
        total += 1
        if ranked[0] in gt_words:
            top1 += 1
        if any(w in gt_words for w in ranked[:5]):
            top5 += 1
        if any(w in gt_words for w in ranked[:10]):
            top10 += 1
        for rank, w in enumerate(ranked):
            if w in gt_words:
                mrr_sum += 1.0 / (rank + 1)
                break

    n = max(total, 1)
    vocab = len(word_list)
    print(f"test images: {total}", flush=True)
    print(f"vocab size:  {vocab}", flush=True)
    print(f"top-1:  {top1}/{total} = {top1/n:.3f}   (random={1/vocab:.4f})", flush=True)
    print(f"top-5:  {top5}/{total} = {top5/n:.3f}   (random={5/vocab:.4f})", flush=True)
    print(f"top-10: {top10}/{total} = {top10/n:.3f}   (random={10/vocab:.4f})", flush=True)
    print(f"MRR:    {mrr_sum/n:.3f}   (random~{1/vocab:.4f})", flush=True)

    # === 4. sentence-level retrieval (standard CLIP eval) ===
    print(f"\n=== sentence-level image<->text retrieval ===", flush=True)
    caps, imgs = [], []
    for cap, img_bytes in test_samples[:200]:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((img_size, img_size))
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        imgs.append(((t - 0.5) / 0.5))
        caps.append(cap)
    imgs_t = torch.stack(imgs).to(device)
    with torch.no_grad():
        img_fs = []
        for i in range(0, len(imgs_t), 64):
            img_fs.append(model.encode_image(imgs_t[i:i+64]))
        img_fs = torch.cat(img_fs, dim=0)
        cap_tok = torch.stack([tokenize(c) for c in caps]).to(device)
        txt_fs = []
        for i in range(0, len(cap_tok), 64):
            txt_fs.append(model.encode_text(cap_tok[i:i+64]))
        txt_fs = torch.cat(txt_fs, dim=0)

    sim_matrix = img_fs @ txt_fs.T  # (N, N)
    N = sim_matrix.size(0)
    # i2t: for each image, rank captions
    i2t_top1 = (sim_matrix.argmax(dim=1) == torch.arange(N, device=device)).float().mean().item()
    t2i_top1 = (sim_matrix.argmax(dim=0) == torch.arange(N, device=device)).float().mean().item()
    # top-5
    i2t_top5 = 0.0
    for i in range(N):
        top5_idx = sim_matrix[i].topk(5).indices
        if i in top5_idx:
            i2t_top5 += 1
    print(f"i2t R@1: {i2t_top1:.3f}  R@5: {i2t_top5/N:.3f}  (random R@1={1/N:.4f})", flush=True)
    print(f"t2i R@1: {t2i_top1:.3f}", flush=True)

    # qualitative
    print("\n=== qualitative ===", flush=True)
    for i in range(min(10, len(caps))):
        top5_idx = sim_matrix[i].topk(5).indices.tolist()
        print(f"  img{i} cap: {caps[i][:70]}", flush=True)
        print(f"    best match: {caps[top5_idx[0]][:70]}", flush=True)


if __name__ == "__main__":
    main()
