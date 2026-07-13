#!/bin/bash
# Overnight training plan (15h budget on 5xH800)
# Usage: nohup bash run_overnight.sh > overnight.log 2>&1 &
# Or in tmux: bash run_overnight.sh

cd /home/jovyan/h800fast/wangzekai/screen-jepa || exit 1
git pull

TAR=/home/jovyan/h800fast/wangzekai/recap-datacomp-384-1M
RDZV="--rdzv-endpoint 127.0.0.1:29500"
TORCHRUN="torchrun --nproc_per_node=5 $RDZV"
mkdir -p outputs

echo "============================================================"
echo "OVERNIGHT TRAINING START: $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────
# Exp1: SlotJEPA base (16 slots, 3 iters) — core experiment
# ─────────────────────────────────────────────────────────
echo "=== $(date) Exp1: SlotJEPA base (16 slots) ==="
$TORCHRUN train_slot_jepa.py \
  --tar_dir $TAR --num_tars 81 --epochs 100 --batch 256 --workers 16 \
  --hidden 768 --layers 12 --heads 12 --pred_depth 4 \
  --num_slots 16 --slot_iters 3 \
  --w_mse 0.3 --w_clip 1.0 --lam 0.1 --lr 3e-4 --log_every 100 \
  --out outputs/slot_base 2>&1 | tee outputs/slot_base.log
echo "=== $(date) Probe slot_base ==="
CUDA_VISIBLE_DEVICES=0 python probe_slot.py \
  --ckpt outputs/slot_base/epoch99.pt --tar_dir $TAR --blank_text 1 \
  2>&1 | tee outputs/slot_base_probe.log

# ─────────────────────────────────────────────────────────
# Exp2: SlotJEPA strong CLIP (w_clip=2.0, w_mse=0.1)
#   Less pixel-prediction weight, more semantic alignment
# ─────────────────────────────────────────────────────────
echo "=== $(date) Exp2: SlotJEPA strong CLIP (w_clip=2.0) ==="
$TORCHRUN train_slot_jepa.py \
  --tar_dir $TAR --num_tars 81 --epochs 100 --batch 256 --workers 16 \
  --hidden 768 --layers 12 --heads 12 --pred_depth 4 \
  --num_slots 16 --slot_iters 3 \
  --w_mse 0.1 --w_clip 2.0 --lam 0.1 --lr 3e-4 --log_every 100 \
  --out outputs/slot_strongclip 2>&1 | tee outputs/slot_strongclip.log
echo "=== $(date) Probe slot_strongclip ==="
CUDA_VISIBLE_DEVICES=0 python probe_slot.py \
  --ckpt outputs/slot_strongclip/epoch99.pt --tar_dir $TAR --blank_text 1 \
  2>&1 | tee outputs/slot_strongclip_probe.log

# ─────────────────────────────────────────────────────────
# Exp3: SlotJEPA 32 slots — more concept capacity
# ─────────────────────────────────────────────────────────
echo "=== $(date) Exp3: SlotJEPA 32 slots ==="
$TORCHRUN train_slot_jepa.py \
  --tar_dir $TAR --num_tars 81 --epochs 100 --batch 256 --workers 16 \
  --hidden 768 --layers 12 --heads 12 --pred_depth 4 \
  --num_slots 32 --slot_iters 3 \
  --w_mse 0.3 --w_clip 1.0 --lam 0.1 --lr 3e-4 --log_every 100 \
  --out outputs/slot32 2>&1 | tee outputs/slot32.log
echo "=== $(date) Probe slot32 ==="
CUDA_VISIBLE_DEVICES=0 python probe_slot.py \
  --ckpt outputs/slot32/epoch99.pt --tar_dir $TAR --blank_text 1 \
  2>&1 | tee outputs/slot32_probe.log

# ─────────────────────────────────────────────────────────
# Exp4: CLIP 12-layer text encoder — stronger baseline
# ─────────────────────────────────────────────────────────
echo "=== $(date) Exp4: CLIP 12-layer text ==="
$TORCHRUN train_clip.py \
  --tar_dir $TAR --num_tars 81 --epochs 100 --batch 512 --workers 16 \
  --hidden 768 --text_layers 12 --text_heads 12 \
  --lr 5e-4 --log_every 100 \
  --out outputs/clip_12L 2>&1 | tee outputs/clip_12L.log
echo "=== $(date) Probe clip_12L ==="
CUDA_VISIBLE_DEVICES=0 python probe_clip.py \
  --ckpt outputs/clip_12L/epoch99.pt --tar_dir $TAR \
  2>&1 | tee outputs/clip_12L_probe.log

# ─────────────────────────────────────────────────────────
# Exp5: SlotJEPA large model (hidden=1024) — more capacity
# ─────────────────────────────────────────────────────────
echo "=== $(date) Exp5: SlotJEPA hidden=1024 ==="
$TORCHRUN train_slot_jepa.py \
  --tar_dir $TAR --num_tars 81 --epochs 100 --batch 192 --workers 16 \
  --hidden 1024 --layers 12 --heads 16 --pred_depth 4 \
  --num_slots 16 --slot_iters 3 \
  --w_mse 0.3 --w_clip 1.0 --lam 0.1 --lr 3e-4 --log_every 100 \
  --out outputs/slot_big 2>&1 | tee outputs/slot_big.log
echo "=== $(date) Probe slot_big ==="
CUDA_VISIBLE_DEVICES=0 python probe_slot.py \
  --ckpt outputs/slot_big/epoch99.pt --tar_dir $TAR --blank_text 1 \
  2>&1 | tee outputs/slot_big_probe.log

echo "============================================================"
echo "ALL EXPERIMENTS DONE: $(date)"
echo "============================================================"
echo "Results summary:"
echo "--- SlotJEPA base ---";     grep -E "top-5|MRR" outputs/slot_base_probe.log 2>/dev/null
echo "--- SlotJEPA strong ---";   grep -E "top-5|MRR" outputs/slot_strongclip_probe.log 2>/dev/null
echo "--- SlotJEPA 32 ---";       grep -E "top-5|MRR" outputs/slot32_probe.log 2>/dev/null
echo "--- CLIP 12L ---";          grep -E "top-5|MRR" outputs/clip_12L_probe.log 2>/dev/null
echo "--- SlotJEPA big ---";      grep -E "top-5|MRR" outputs/slot_big_probe.log 2>/dev/null
echo "--- PREVIOUS (for reference) ---"
echo "Cross-modal JEPA (no slot): top-5=13.7%"
echo "CLIP 8-layer: top-5=51.3%"
