# Project structure

## Quy ước

- Tên thư mục và tệp mới dùng `lower_snake_case`; tên lớp Python dùng
  `PascalCase`, hàm và biến dùng `snake_case`.
- `src` chỉ chứa mã lõi có thể nhập lại; chương trình thực thi nằm trong
  `scripts` hoặc `apps`.
- `data` là dữ liệu đầu vào; `artifacts` là đầu ra có thể sinh lại; `archive`
  không được dùng làm input mặc định.
- Mọi lựa chọn ngưỡng/policy dùng Validation. Test chỉ dùng để báo cáo.

## Cây thư mục

```text
TTTN/
├── apps/
│   ├── dataset_review/        Review, sửa mask, lưu tiến độ và export dataset
│   └── web_demo/              React UI + FastAPI inference
├── src/threecad_segmentation/
│   ├── *_r18.py, *_b0.py      Kiến trúc U-Net và SegFormer
│   ├── vmamba_t.py            Adapter VMamba-T nhị phân
│   ├── train_common.py        Vòng lặp train dùng chung
│   ├── fullres_eval.py        Suy luận ảnh đầy đủ và metric
│   ├── decision_policy.py     Logic không gian/connected components
│   ├── adaptive_component_policy.py
│   └── third_party/VMamba/    Mã vendor; không chứa `.git` lồng
├── scripts/
│   ├── data/                  Data audit
│   ├── demo/                  Khởi động demo
│   ├── evaluation/            Validation/Test từng model
│   ├── experiments/           Policy, spatial ensemble và probability cache
│   ├── reporting/             Tổng hợp bảng/biểu đồ/audit Test
│   ├── setup/                 Cài runtime VMamba trên Kaggle
│   ├── training/              Entry point train
│   └── verification/          Protocol và kiểm tra bàn giao
├── tests/ml/                  Unit/smoke tests cho mã lõi
├── data/3cad_ani/             Ảnh, mask, frozen splits và dataset audit
├── artifacts/
│   ├── checkpoints/final/     Ba best checkpoint cuối
│   ├── experiments/           Cache/kết quả chạy trung gian
│   ├── packages/kaggle/       Gói Kaggle đã đóng băng
│   ├── reports/final/         Bảng, audit và visualization cuối
│   └── verification/          Kết quả protocol preflight
├── experiments/notebooks/     Notebook audit và pipeline Kaggle
├── runtime/wheels/vmamba/     Wheel phụ thuộc GPU/Python cụ thể
├── docs/                      Protocol và hướng dẫn
├── submission/                Hồ sơ nộp bài
└── archive/                   Mã không thuộc luồng chạy chính
```

## Đường dẫn ổn định

Ba launcher ở gốc (`run_review.ps1`, `run_demo.ps1`, `verify.ps1`) là giao diện
ổn định cho người dùng. Mã Python dùng `src/threecad_segmentation/project_paths.py`
hoặc tính `repo_root` từ vị trí tệp, không phụ thuộc ổ `E:`.
