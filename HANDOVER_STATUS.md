# Trạng thái bàn giao — 04/11/2026

**Docker và giao diện v3.5:** gỡ route/trang Nghiên cứu; chuyển CPU/GPU/RAM/track
khỏi trang kiểm định sang `/performance`, lấy mẫu thật mỗi giây và vẽ biểu đồ.
Một image chứa web, xử lý ảnh/video, ONNX và worker TensorRT, tự tạo engine
theo GPU/driver/TensorRT và lưu dữ liệu trong volume. Hướng dẫn
`docs/STRAW_DOCKER_DEMO.md`; hồ sơ `artifacts/verification/upgrade_v3_5/`.

**Sửa chất lượng ảnh v3.4:** nguồn ảnh 1920×1080, preview riêng 960×540;
giữ nguyên ngưỡng nét 18. Có kiểm thử ảnh thật sự mờ/tối và truy nguyên độ
phân giải nguồn/crop trong lịch sử. Gói mới `artifacts/releases/v3_4/`;
hồ sơ `artifacts/verification/upgrade_v3_4/`, hướng dẫn
`docs/STRAW_PHOTO_QUALITY_V3_4.md`.

**Sửa tracking v3.3 — 04/11/2026:** kiểm tra trọn 39 ảnh ở chế độ một/ba sản
phẩm: đủ 39 bản ghi, không thiếu/lặp/crop thiếu; phân biệt review chất lượng
với mất track. Gói mới ở `artifacts/releases/v3_3/`. Hồ sơ ở
`artifacts/verification/upgrade_v3_3/` và `docs/STRAW_TRACKING_FIX_V3_3.md`.

**Bổ sung v3.2:** theo dõi, cắt ảnh và phân loại từng sản phẩm khi 2–3 sản phẩm
cùng nằm trong ROI; giữ các mask và telemetry. Đã bỏ nhánh xử lý nhập nhóm và
chế độ mô phỏng tương ứng. Giới hạn 50 ảnh/lượt. Gói mới ở
`artifacts/releases/v3_2/`, bằng chứng mới ở `artifacts/verification/upgrade_v3_2/`.
Xem `docs/STRAW_UPGRADE_V3.md`; hồ sơ các phiên bản cũ được giữ riêng.

**Bổ sung v2:** đã có đường video/RTSP dùng giải mã GPU, tracking cho camera
cố định, chọn frame/kiểm tra chất lượng, lịch sử SQLite và dashboard nghiên cứu.
Đã kiểm chứng bằng video tổng hợp và RTSP replay; chưa có camera vật lý.
Hồ sơ và phạm vi: `docs/STRAW_UPGRADE_V2.md`. Gói mới lưu riêng dưới
`artifacts/releases/v2/`.

Phạm vi gói: mã nguồn ống hút, demo TensorRT trong môi trường DeepStream,
checkpoint/ONNX/engine đúng cấu hình `b1_stream_512_pose_v1`, ảnh mẫu, clip
replay, báo cáo và bằng chứng chọn lọc. Trạng thái kiểm tra thực tế nằm trong
`delivery_verification.json` bên cạnh ZIP và các log tại `artifacts/verification/release/`.

| Hạng mục | Tình trạng |
|---|---|
| Mã nguồn và cấu hình demo | Có trong gói; nguồn từ working tree, giữ nguyên kết quả nghiên cứu gốc |
| Checkpoint, ONNX và engine | Có SHA-256 và kiểm tra liên kết model/config/class order |
| Cài/chạy/dừng/kiểm tra | `setup_demo.ps1`, `run_b1_conveyor.ps1`, `stop_demo.ps1`, `verify_demo.ps1` |
| Kiểm tra bản giải nén | Ghi nhận kết quả thực trong `delivery_verification.json` |
| Image SDK và WSL | Cài riêng; gói không chứa môi trường ảo/ổ đĩa WSL |
| Thực nghiệm B0–B6 | Hồ sơ 21 run và bảng kết quả đi kèm bản đầy đủ; không gộp với engine demo |
| Báo cáo Word/PDF | Bản hiện có được đưa vào `submission/01_report/`; chưa phải xác nhận đã duyệt |
| Slide, video bảo vệ và chữ ký | Chưa có bản hoàn chỉnh được xác nhận cho đề tài ống hút |
| Nguồn/quyền sử dụng dataset và rà nhãn/nhóm | Cần hồ sơ xác nhận thật |
| Video camera độc lập / vận hành thực | Chưa được kiểm chứng đầy đủ |

`freeze.json` nghiên cứu hiện chứa reviewer/notes dạng mẫu dù các cờ review
được bật. Gói giữ nguyên hồ sơ gốc để truy nguyên; không coi trạng thái tự động
`READY_FOR_INTERNAL_THESIS_ANALYSIS` là xác nhận con người đã rà dữ liệu hoặc
phê duyệt của trường. Cần thay bằng biên bản thực qua quy trình nghiên cứu,
không sửa lại checkpoint/Test để làm đẹp kết quả.

Không đưa báo cáo antifake news hoặc hồ sơ nhôm ANI vào gói ống hút.
Ngày tạo gói không phải ngày huấn luyện hay ngày sinh kết quả thử nghiệm.
