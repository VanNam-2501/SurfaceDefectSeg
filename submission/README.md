# Hồ sơ chuẩn bị nộp đồ án

Thư mục này là bảng điều khiển hồ sơ nộp. Nó không sao chép dataset, checkpoint
hoặc kết quả lớn; thay vào đó ghi rõ nguồn chuẩn và trạng thái từng sản phẩm.

## Bắt đầu từ đây

1. Đọc `CHECKLIST.md`.
2. Hoàn thiện báo cáo trong `01_report/`.
3. Hoàn thiện slide trong `02_slides/`.
4. Thiết lập Git cho mã nguồn theo `03_source/README.md`.
5. Quay video theo `04_demo_video/README.md`.
6. Bổ sung nhật ký tại `05_logs/` sau mỗi phiên làm việc.
7. Đưa bản scan xác nhận đơn vị vào `06_company_confirmation/`.
8. Chạy `..\verify.ps1 -IncludeWeb` và lưu đầu ra vào
   `07_evidence/` trước khi đóng gói.

## Nguồn chuẩn

| Thành phần | Đường dẫn |
|---|---|
| Nguồn gốc dataset 3CAD ANI | `07_evidence/DATASET_PROVENANCE.md` |
| Mã ML | `../src/threecad_segmentation/` |
| Demo web | `../apps/web_demo/` |
| Công cụ review data | `../apps/dataset_review/` |
| Checkpoint cuối | `../artifacts/checkpoints/final/` |
| Kết quả cuối | `../artifacts/reports/final/` |
| Dashboard biểu đồ | `../artifacts/reports/final/visualizations/index.html` |

Đường dẫn trên là cấu trúc chuẩn sau migration 2026-08-19. Xem
`../docs/MIGRATION_MANIFEST.md` khi đối chiếu log/tài liệu lịch sử. Các file ZIP
Kaggle cũ được kiểm kê trong `07_evidence/KAGGLE_ARCHIVE_INVENTORY.md`.
