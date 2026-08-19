# Nhật ký phát triển

Điền mỗi dòng ngay trong ngày làm việc. Không thay log này bằng AI Development
Log; hai tài liệu có mục đích khác nhau.

| Ngày | Thời lượng | Mục tiêu | Việc đã làm | File/Module | Kết quả/Test | Vấn đề | Quyết định | Commit |
|---|---:|---|---|---|---|---|---|---|
| 2026-08-15 | CẦN ĐIỀN | Review dữ liệu | Xây dựng và hiệu chỉnh review tool | `apps/dataset_review/` | CẦN ĐÍNH KÈM | UI phải xem nhanh nhiều ảnh | Lưu tiến độ, review queue | `d91f6fd` (baseline) |
| 2026-08-16 | CẦN ĐIỀN | Training/evaluation | Chuẩn hóa pipeline và Kaggle | ML pipeline | Protocol PASS | VRAM khác nhau | Chạy từng model | `d91f6fd` (baseline) |
| 2026-08-17 | CẦN ĐIỀN | Giảm false alarm | Adaptive và learned verifier | Decision modules | Unit test PASS | FPR/FNR trade-off | Chọn bằng Validation | `d91f6fd` (baseline) |
| 2026-08-18 | CẦN ĐIỀN | Kết quả cuối | Tổng hợp thí nghiệm, audit, biểu đồ | `artifacts/reports/final/` | 20 PNG/19 SVG | Scope quá rộng | Loại E1/E6/E9/E10/E11 | `d91f6fd` (baseline) |
| 2026-08-19 | CẦN ĐIỀN | Hồ sơ nộp | Cleanup, compliance audit | `submission/` | Verify PASS | Chưa có Git history | Không giả tạo lịch sử | `d91f6fd` (baseline) |
| 2026-08-19 | CẦN ĐIỀN | Chuẩn hóa repository | Tách `src/apps/scripts/data/artifacts`, sửa path và test | Toàn repository | Protocol + 15 Python test + 2 web test PASS | `.git` vendor và thư mục bị lock | Hash trước cleanup, một Git root | `d91f6fd` |
