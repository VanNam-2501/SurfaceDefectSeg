# 3CAD-ANI Aluminum Surface Defect Segmentation

Dự án thực tập tốt nghiệp về phân đoạn khuyết tật bề mặt nhôm trên tập con
**ANI (Aluminum New iPad) của 3CAD**. Hệ thống gồm U-Net ResNet-18,
SegFormer-B0, VMamba-T, lựa chọn ngưỡng chỉ trên Validation, đánh giá Test cố
định, logic giảm dương tính giả và hai ứng dụng review/demo.

## Bắt đầu nhanh

```powershell
# Kiểm tra mã, protocol, model tests và các artifact bắt buộc
.\verify.ps1

# Review và sửa nhãn/mask
.\run_review.ps1

# Demo dự báo U-Net + SegFormer + VMamba
.\run_demo.ps1
```

Thêm `-IncludeWeb` vào `verify.ps1` để chạy cả test frontend.

## Cấu trúc chính

```text
apps/          ứng dụng Dataset Review và web demo
src/           model, dataset, loss và logic quyết định dùng lại được
scripts/       lệnh train, evaluation, experiment, reporting và verification
tests/         kiểm thử mã ML
data/          dữ liệu 3CAD-ANI cục bộ (không đưa vào Git)
artifacts/     checkpoint, prediction, kết quả và báo cáo sinh ra
experiments/   notebook audit dữ liệu và pipeline Kaggle
docs/          protocol, hướng dẫn và hồ sơ cấu trúc
runtime/       wheel nhị phân VMamba theo môi trường
submission/    hồ sơ nộp bài và hồ sơ sử dụng AI
archive/       dữ liệu cũ và ứng viên cleanup có thể khôi phục
```

Chi tiết trách nhiệm từng thư mục nằm tại
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Lịch sử đổi đường dẫn
nằm tại [`docs/MIGRATION_MANIFEST.md`](docs/MIGRATION_MANIFEST.md).

## Quy trình thí nghiệm đúng

1. Kiểm tra split và rò rỉ dữ liệu bằng `scripts/verification/check_protocol.py`.
2. Train từng kiến trúc độc lập nếu cần; checkpoint tốt nhất chọn theo
   Validation, không theo Test.
3. Chọn threshold/policy trên Validation.
4. Khóa threshold/policy và đánh giá đúng một lần trên Test.
5. Dùng toàn bộ Train/Validation/Test chỉ cho data audit riêng, không dùng các
   kết quả đó để điều chỉnh lại bảng Test.

Chạy lại toàn bộ logic ba model từ probability cache:

```powershell
.\scripts\experiments\run_three_model_experiments.ps1
```

Train lại một model:

```powershell
.\scripts\training\retrain_models.ps1 -Model unet -RunName unet_seed42_v2
```

## Kết quả và hồ sơ nộp

- Dashboard kết quả: `artifacts/reports/final/visualizations/index.html`.
- Checkpoint cuối: `artifacts/checkpoints/final/`.
- Checklist nộp bài: `submission/CHECKLIST.md`.
- AI Development Log: `submission/05_logs/AI_DEVELOPMENT_LOG.md`.
- Nguồn gốc dataset: `submission/07_evidence/DATASET_PROVENANCE.md`.

Dataset, checkpoint, artifact sinh ra, môi trường ảo và `archive` được loại khỏi
Git. Mã nguồn, cấu hình, tài liệu, notebook và lịch sử commit phải được lưu đầy
đủ để có thể giải thích và tái tạo quy trình.
