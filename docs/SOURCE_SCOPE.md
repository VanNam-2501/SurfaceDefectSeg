# Phạm vi mã nguồn nhánh ống hút

Nhánh `straw-defect-classification` chứa mã và tài liệu phục vụ phân loại
khuyết tật ống hút trên hộp đồ uống. Dự án phân đoạn bề mặt nhôm 3CAD-ANI
được giữ tại [main](https://github.com/VanNam-2501/SurfaceDefectSeg/tree/main).

| Giữ trong Git | Lý do |
|---|---|
| `src/straw_global_local/`, `src/straw_defect_cls/`, `src/inspection_research/` | Huấn luyện, protocol, đánh giá, hiệu chuẩn và suy luận ống hút |
| `src/straw_defect_cls/third_party/VMamba/` | Phụ thuộc tùy chọn của classifier VMamba; giữ mã nguồn và license gốc |
| `apps/stream_demo/`, `docker/demo/`, script triển khai | Luồng ảnh/video/RTSP, tracking, TensorRT, Docker và kiểm tra runtime |
| `apps/web_demo/backend/b1_*`, `hierarchical_app.py` | Hai API ống hút trước đây còn được script đối chiếu và kiểm thử sử dụng |
| Cấu hình, requirements, notebook và script dữ liệu/nghiên cứu | Tái tạo quy trình; không chứa ảnh hoặc trọng số |
| Kiểm thử và công cụ đóng gói | Kiểm tra protocol, model, tracking và gói bàn giao |
| `docs/`, `docs/assets/straw_overview/` | Hướng dẫn, nguồn báo cáo và hình tổng quan chọn lọc |
| `submission/tables/`, hình tổng hợp nhỏ | Bằng chứng lịch sử của pipeline Full5/cascade; không gộp với B0–B6 |
| Script tạo báo cáo ống hút | Tái tạo báo cáo từ nguồn Markdown |

Không theo dõi báo cáo antifake news, mã/ứng dụng/hồ sơ nhôm, archive,
`experiments/runs/`, `_report_work/`, ảnh occlusion dung lượng lớn,
Word/PDF sinh ra, dataset, checkpoint, ONNX, engine, cache, log, wheel,
ZIP hoặc môi trường cục bộ. Các file đã có trên máy được giữ nguyên;
`.gitignore` ngăn việc thêm lại vào nhánh này.

Word/PDF, model, media mẫu và kết quả đầy đủ được cung cấp qua gói bàn giao
riêng theo `DEEPSTREAM_HANDOVER.md` và `HANDOVER_STATUS.md`. Bản clone mã
nguồn không có sẵn engine hoặc dataset để chạy inference thật.

Trước khi push: kiểm tra `git diff --cached --stat`, kiểm thử từ bản export
chỉ chứa file Git theo dõi, kiểm tra source ZIP và xác nhận chỉ cập nhật
nhánh ống hút. Việc dọn file không viết lại lịch sử Git; các commit cũ vẫn
chứa snapshot trước khi dọn.
