# Aluminum Surface Defect Segmentation — Colab

This is the final-training package after smoke testing all three architectures.
It adds positive-only Dice, full-resolution Val loss curves, ROC/PR/confusion
matrix outputs, CUDA timing warm-up, early stopping and detailed timing/VRAM logs.

## 1. Required Colab layout

Keep the project at:

```text
/content/TTTN/
```

and dataset at:

```text
/content/TTTN/data/3cad_ani/
```

Frozen split files:

```text
/content/TTTN/data/3cad_ani/dataset_audit/splits/train.csv
/content/TTTN/data/3cad_ani/dataset_audit/splits/val.csv
/content/TTTN/data/3cad_ani/dataset_audit/splits/test.csv
```

For VMamba, keep the wheel you already built:

```text
/content/drive/MyDrive/TTTN/wheels/
mamba_ssm-2.3.2.post1+cu128torch2.11sm75-cp312-cp312-linux_x86_64.whl
```

## 2. Common setup

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/TTTN
```

```bash
!pip install -q -r requirements_colab.txt
!python check_protocol.py
```

## 3. U-Net/ResNet18 final

```bash
!python train_unet.py \
  --epochs 50 --batch-size 2 --grad-accum 2 \
  --augmentation photometric --run-name main_seed42

!python evaluate_model.py \
  --model unet \
  --checkpoint results/unet_r18/main_seed42/checkpoints/best.pt \
  --split val --warmup-batches 5

!python evaluate_model.py \
  --model unet \
  --checkpoint results/unet_r18/main_seed42/checkpoints/best.pt \
  --split test --warmup-batches 5
```

### Optional E1 resize baseline

```bash
!python train_unet.py \
  --epochs 50 --batch-size 2 --grad-accum 2 \
  --augmentation photometric --data-mode resize --run-name e1_resize_seed42

!python evaluate_model.py --model unet \
  --checkpoint results/unet_r18/e1_resize_seed42/checkpoints/best.pt --split val
!python evaluate_model.py --model unet \
  --checkpoint results/unet_r18/e1_resize_seed42/checkpoints/best.pt --split test
```

## 4. SegFormer-B0 final

Uses ImageNet-only `nvidia/mit-b0` encoder pretraining; do not reuse the old
ADE20K-finetuned smoke checkpoint.

```bash
!python train_segformer.py \
  --epochs 50 --batch-size 2 --grad-accum 2 \
  --augmentation photometric --run-name main_seed42

!python evaluate_model.py --model segformer \
  --checkpoint results/segformer_b0/main_seed42/checkpoints/best.pt --split val
!python evaluate_model.py --model segformer \
  --checkpoint results/segformer_b0/main_seed42/checkpoints/best.pt --split test
```

## 5. VMamba-T final

Setup reproduces the working T4 configuration you already verified.

```bash
!PROJECT_ROOT=/content/TTTN bash setup_vmamba_colab.sh
!python test_vmamba_runtime.py
```

Do not continue unless the output includes:

```text
selective_scan_cuda: ...so
WITH_SELECTIVESCAN_MAMBA = True
VMAMBA MODEL TEST: PASS
```

Then:

```bash
!python train_vmamba.py \
  --epochs 50 --batch-size 1 --grad-accum 4 \
  --augmentation photometric --run-name main_seed42

!python evaluate_model.py --model vmamba \
  --checkpoint results/vmamba_t_s2l5/main_seed42/checkpoints/best.pt --split val
!python evaluate_model.py --model vmamba \
  --checkpoint results/vmamba_t_s2l5/main_seed42/checkpoints/best.pt --split test
```

## 6. Compare three models

```bash
!python compare_models.py
```

Optional E9 after seeds 42/123/2026 are complete:

```bash
!python compare_seeds.py
```

## 7. What to inspect after each run

Training:

```text
curves/learning_curve_loss.png
curves/learning_curve_dice.png
curves/loss_components.png
curves/learning_rate.png
curves/epoch_time.png
curves/vram.png
monitor/epoch_XXX/*.png
training_history.csv
training_summary.json
```

Validation:

```text
validation/threshold_scan.csv
validation/threshold_selection.png
validation/selected_threshold.json
validation/figures/image_roc_curve.png
validation/figures/image_pr_curve.png
validation/figures/confusion_matrix.png
```

Test:

```text
test/main_metrics.csv
test/defect_size_metrics.csv
test/defect_group_metrics.csv
test/multi_region_metrics.csv
test/figures/image_roc_curve.png
test/figures/image_pr_curve.png
test/figures/confusion_matrix.png
test/qualitative/*.png
```

Read `EXPERIMENT_PROTOCOL.md` before starting final runs.
