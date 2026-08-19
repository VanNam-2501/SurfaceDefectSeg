# Train trên Kaggle

File này là quy trình Kaggle cho U-Net/ResNet-18, SegFormer-B0 và VMamba-T.
Metric segmentation ở Train/Val/Test vẫn là metric gốc của model. Logic
Pass/Review/Defect chỉ dùng sau này ở web demo, không được dùng để chọn
checkpoint hoặc tinh chỉnh Test.

## 1. Chuẩn bị Kaggle Notebook

Trong Notebook settings, bật **GPU** và **Internet**. Internet chỉ cần để cài
package, clone VMamba và tải ImageNet pretraining lần đầu. Kaggle input là
read-only; mọi checkpoint và kết quả phải ghi vào `/kaggle/working`.

Các đầu vào đã được chuẩn bị tại:

1. `artifacts/packages/kaggle/threecad_ani_selected_experiments_2026-08-18.zip`:
   code đánh giá, checkpoint và dataset 3CAD-ANI tại `data/3cad_ani`.
2. `runtime/wheels/vmamba/mamba_ssm-2.3.2.post1+cu128torch2.11sm75-cp312-cp312-linux_x86_64.whl`:
   wheel VMamba.

Tạo hai Kaggle Dataset riêng từ hai file trên và Add Input cả hai vào Notebook.
Giữ chúng ở chế độ Private nếu dữ liệu không được phép công khai.

Gói project-data hiện chứa dataset gốc đã audit, chưa chứa thay đổi từ Data
Review Studio vì hiện chưa có bản export trong `apps/dataset_review/exports`.
Sau khi review xong, hãy upload thêm chính thư mục `training_dataset` đã export
và đặt `DATASET_ROOT` tới input đó.

VMamba cần thêm input wheel riêng chứa file đã chuẩn bị:

```text
mamba_ssm-2.3.2.post1+cu128torch2.11sm75-cp312-cp312-linux_x86_64.whl
```

Wheel này chỉ dùng cho Python 3.12, PyTorch 2.11, CUDA 12.8 và Tesla T4/sm75.
Trong Kaggle accelerator phải chọn T4; không dùng P100.

## 2. Kiểm tra runtime trước khi import Torch

Chạy cell này ở đầu Notebook:

```python
import sys, subprocess
print("Python:", sys.version)
subprocess.run(["nvidia-smi"], check=True)
```

Python phải là `3.12.x` và GPU phải là Tesla T4. Sau đó cài đúng PyTorch mà
wheel đã được build cùng:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
    "--index-url", "https://download.pytorch.org/whl/cu128",
], check=True)
```

Nếu cell thực sự thay đổi phiên bản Torch, hãy **Restart Session** một lần rồi
chạy lại từ đầu. Không import `torch` trước khi restart.

## 3. Cell tìm input và copy code vào vùng ghi được

Cell sau tự tìm project dù Kaggle đã giải nén ZIP hay vẫn giữ nguyên file ZIP:

```python
from pathlib import Path
import shutil, zipfile

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
project_matches = list(INPUT.rglob("scripts/training/train_on_kaggle.py"))

if project_matches:
    # Kaggle đã mở nội dung archive thành input read-only. Chỉ copy code;
    # dataset vẫn được đọc trực tiếp từ /kaggle/input.
    PROJECT_INPUT = project_matches[0].parents[2]
    PROJECT = WORKING / "threecad_ani_project"
    shutil.copytree(
        PROJECT_INPUT,
        PROJECT,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("data", "artifacts", "archive", ".venv", "node_modules", "__pycache__"),
    )
    DATASET_ROOT = PROJECT_INPUT / "data" / "3cad_ani"
else:
    # Trường hợp Kaggle giữ file ZIP: giải nén vào vùng ghi được.
    archives = list(INPUT.rglob("threecad_ani_selected_experiments_*.zip"))
    assert archives, "Chưa Add Input project-data vào Notebook"
    extracted = WORKING / "tttn_project_data"
    with zipfile.ZipFile(archives[0]) as zf:
        zf.extractall(extracted)
    project_matches = list(extracted.rglob("scripts/training/train_on_kaggle.py"))
    assert project_matches, "ZIP project không hợp lệ"
    PROJECT_INPUT = project_matches[0].parents[2]
    PROJECT = WORKING / "threecad_ani_project"
    shutil.copytree(PROJECT_INPUT, PROJECT, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("data", "artifacts", "archive", ".venv", "node_modules", "__pycache__"))
    DATASET_ROOT = PROJECT_INPUT / "data" / "3cad_ani"

# Nếu đã Add Input dataset sau review, thay DATASET_ROOT bằng đường dẫn tìm được:
# reviewed = list(INPUT.rglob("training_dataset/dataset_audit/splits/train.csv"))
# if reviewed: DATASET_ROOT = reviewed[0].parents[2]

assert (PROJECT / "scripts/training/train_on_kaggle.py").is_file(), "Sai PROJECT_INPUT"
assert (DATASET_ROOT / "dataset_audit/splits/train.csv").is_file(), "Sai DATASET_ROOT"
print("Project:", PROJECT)
print("Dataset:", DATASET_ROOT)
```

Không copy dataset vào `working`: code đọc trực tiếp từ `/kaggle/input`, giúp
tiết kiệm dung lượng. Kaggle lưu output của notebook trong `/kaggle/working`.

## 4. Cell cài package và kiểm tra data

```python
import os, subprocess, sys

os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements/ml-kaggle.txt"], check=True)
subprocess.run([
    sys.executable, "scripts/verification/check_protocol.py",
    "--dataset-root", str(DATASET_ROOT),
    "--train-csv", str(DATASET_ROOT / "dataset_audit/splits/train.csv"),
    "--val-csv", str(DATASET_ROOT / "dataset_audit/splits/val.csv"),
    "--test-csv", str(DATASET_ROOT / "dataset_audit/splits/test.csv"),
    "--save", "/kaggle/working/protocol_check.json",
], check=True)
```

Phải thấy `PROTOCOL CHECK: PASS` trước khi train.

## 5. Train U-Net và SegFormer

Launcher tự làm toàn bộ: train → chọn threshold bằng **Validation** → test với
threshold đã khóa. Không dùng Test để điều chỉnh threshold.

```python
RUN = "cleaned_v1_seed42"
OUT = "/kaggle/working/results"

subprocess.run([
    sys.executable, "scripts/training/train_on_kaggle.py",
    "--model", "unet",
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", OUT,
    "--run-name", RUN,
    "--epochs", "50",
    "--early-stopping-patience", "8",
    "--early-stopping-min-delta", "0.0001",
], check=True)

subprocess.run([
    sys.executable, "scripts/training/train_on_kaggle.py",
    "--model", "segformer",
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", OUT,
    "--run-name", RUN,
    "--epochs", "50",
    "--early-stopping-patience", "8",
    "--early-stopping-min-delta", "0.0001",
], check=True)
```

Batch mặc định đã được đặt theo model:

- U-Net/SegFormer: `batch-size=2`, `grad-accum=2` → effective batch 4.
- VMamba: `batch-size=1`, `grad-accum=4` → effective batch 4.

Nếu cần override vì thiếu VRAM, thêm `"--batch-size", "1", "--grad-accum", "4"`
vào lệnh của model đó. Các tham số này được lưu vào `config.json` của run.

Checkpoint tốt nhất và toàn bộ metrics có dạng:

```text
/kaggle/working/results/unet_r18/cleaned_v1_seed42/checkpoints/best.pt
/kaggle/working/results/segformer_b0/cleaned_v1_seed42/checkpoints/best.pt
```

## 6. VMamba-T (chỉ sau khi runtime pass)

Trong Kaggle cell, sửa đường dẫn wheel rồi chạy:

```python
os.environ["PROJECT_ROOT"] = str(PROJECT)
from pathlib import Path
wheel_matches = list(Path("/kaggle/input").rglob("mamba_ssm-2.3.2.post1+cu128torch2.11sm75-cp312-cp312-linux_x86_64.whl"))
assert wheel_matches, "Chưa add Kaggle input chứa wheel Mamba"
os.environ["MAMBA_WHEEL"] = str(wheel_matches[0])
subprocess.run(["bash", "scripts/setup/setup_vmamba_kaggle.sh"], check=True)
```

Cell phải kết thúc bằng `VMAMBA MODEL TEST: PASS`. Nếu không pass, **không train
VMamba**; không dùng wheel T4/Colab cũ khi Python, Torch, CUDA hoặc GPU Kaggle
khác với wheel đó.

Khi pass, train VMamba:

```python
subprocess.run([
    sys.executable, "scripts/training/train_on_kaggle.py",
    "--model", "vmamba",
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", OUT,
    "--run-name", RUN,
    "--epochs", "50",
    "--early-stopping-patience", "8",
    "--early-stopping-min-delta", "0.0001",
], check=True)
```

Nếu cả ba runtime đã sẵn sàng từ đầu, có thể dùng một lệnh:

```python
subprocess.run([
    sys.executable, "scripts/training/train_on_kaggle.py",
    "--model", "all",
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", OUT,
    "--run-name", RUN,
], check=True)
```

## 7. Lưu kết quả

Trước khi kết thúc, chọn **Save Version / Commit** để Kaggle lưu toàn bộ
`/kaggle/working/results` thành notebook output. Có thể thêm output này vào
notebook web demo hoặc tải `best.pt`, `validation/selected_threshold.json` và
`test/` về máy.

Nếu phiên có persistence, chỉ file trong `/kaggle/working` được giữ giữa các
session; input luôn là read-only.
