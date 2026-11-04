# Kiểm tra khuyết tật ống hút trên hộp đồ uống

Phạm vi hiện tại: chỉ bộ dữ liệu ống hút hiện có; phân loại 5 trạng thái để
xác định sản phẩm bình thường hoặc lỗi gì. Hướng nghiên cứu ứng viên là CNN
kết hợp đặc trưng toàn ảnh và cục bộ, với đối chứng độ phân giải, attention và
thông tin vị trí. Phạm vi triển khai hiện hành không chưng cất.
Đọc **[kế hoạch nghiên cứu chính](docs/STRAW_RESEARCH_PLAN.md)** để xem câu hỏi,
thiết kế, thí nghiệm, tiêu chí kết luận và phần chưa triển khai.
Đối chiếu tài liệu, thứ tự ưu tiên và thiết kế teacher/student nằm trong
[bản tổng hợp nghiên cứu](docs/STRAW_RESEARCH_SYNTHESIS.md).

Thương hiệu chỉ là metadata; LOBO không còn là mục tiêu/thí nghiệm bắt buộc.
Mô hình B0–B6 và pipeline mới đã triển khai trong `src/straw_global_local/`;
chưa có kết quả xác nhận để kết luận hiệu quả. Đọc
[hướng dẫn chạy](docs/STRAW_GLOBAL_LOCAL_RUNBOOK.md) và
[điều kiện đủ bằng chứng kết luận](docs/STRAW_COMPLETION_REQUIREMENTS.md).
Kết quả kiểm chứng kỹ thuật và pilot nằm trong
[hồ sơ kiểm chứng](docs/STRAW_GLOBAL_LOCAL_VERIFICATION.md).
Notebook riêng: `experiments/notebooks/straw_global_local_kaggle.ipynb`.
Để chạy đủ 21 lượt trên Kaggle với tự dừng theo ngân sách giờ và tiếp tục ở
phiên sau, dùng [gói code Kaggle có resume](docs/KAGGLE_ALL_IN_ONE.md):
`artifacts/packages/straw_kaggle_code.zip`. Gói không chứa dataset/weights;
đọc trực tiếp dataset gắn qua Add Input và tự xác định đường dẫn.
Có thể import một file duy nhất
`artifacts/packages/STRAW_KAGGLE_ALL_EXPERIMENTS.ipynb` đã nhúng toàn bộ code,
không cần upload ZIP source thành Input riêng.
`src/inspection_research/` hiện chứa prototype MixStyle và công cụ toán học
của hướng trước. Pipeline hai tầng `PASS / REVIEW / DEFECT`, cấu hình 72 run
và notebook bên dưới là hạ tầng cũ, chưa chạy kế hoạch mới. Hướng dẫn lịch sử:
[STRAW_THESIS.md](docs/STRAW_THESIS.md).

```powershell
.\verify_straw.ps1
.\run_straw_thesis.ps1 -Action plan
```

Artifact của pipeline lịch sử lưu tại `artifacts/straw_thesis/`; kết quả cũ
không tự động được coi là kết quả khóa luận. Notebook Kaggle của pipeline này:
`experiments/notebooks/straw_thesis_kaggle.ipynb`.

## Pipeline phân đoạn nhôm trước đây

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
