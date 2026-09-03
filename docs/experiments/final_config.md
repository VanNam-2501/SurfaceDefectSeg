# Cấu hình thí nghiệm thực tế từ checkpoint cuối

Tài liệu này phản ánh checkpoint thực tế đã dùng để tạo bảng kết quả cuối. Nó
được ưu tiên hơn các con số batch size trong `EXPERIMENT_PROTOCOL.md`, vốn là
kế hoạch ban đầu trước khi điều chỉnh theo VRAM Kaggle.

## Cấu hình chung

- Dataset: **3CAD — Aluminum New Ipad (ANI)**. Bản gốc cục bộ được audit có
  7.169 ảnh; frozen experimental set dùng 7.168 ảnh sau khi bỏ một ảnh trùng
  nguồn, gồm Train 5.733, Validation 718, Test 717.
- Seed: 42.
- Patch: 512 × 512; stride đánh giá 256.
- Augmentation: `photometric`.
- Encoder LR: `1e-5`; decoder LR: `1e-4`.
- Weight decay: `1e-4`.
- Early stopping patience: 8; min delta: `1e-4`.
- LR scheduler patience: 3; factor: 0,5.
- Threshold segmentation chỉ được chọn trên Validation.

## Cấu hình thực tế theo model

| Model | Batch | Grad accumulation | Effective batch | Epoch tối đa | Best epoch | Best Val Positive Dice@0.5 | Threshold cuối |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net/ResNet18 | 38 | 1 | 38 | 50 | 48 | 0,702160 | 0,49 |
| SegFormer-B0 | 16 | 2 | 32 | 50 | 50 | 0,691986 | 0,66 |
| VMamba-T | 16 | 1 | 16 | 25 | 2có3 | 0,768361 | 0,51 |

Nguồn cấu hình: trường `config`, `best_epoch` và `best_metric` bên trong:

- `artifacts/checkpoints/final/unet_best.pt`;
- `artifacts/checkpoints/final/segformer_best.pt`;
- `artifacts/checkpoints/final/vmamba_best.pt`.

## Scope báo cáo đã chốt

- Bao gồm: E0, E2, E3, E4, E5, E7, E8.
- Không thuộc scope hiện tại: E1, E6, E9, E10, E11.
- Logic quyết định và audit dữ liệu là phần bổ sung sau segmentation baseline.

Scope máy đọc được nằm tại
`artifacts/reports/final/thesis_evaluation_report/scope.json`.

## Lưu ý phương pháp

Batch size khác nhau vì giới hạn VRAM và tốc độ từng kiến trúc. Vì vậy không
được tuyên bố ba model có effective batch bằng nhau. So sánh E2 vẫn dùng cùng
split, preprocessing, loss, seed và quy tắc Validation/Test; khác biệt batch
phải được công khai trong báo cáo như một giới hạn thực nghiệm.
