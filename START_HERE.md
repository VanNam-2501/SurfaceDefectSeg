# Bàn giao dự án ống hút — DeepStream / TensorRT

**v3.5:** ưu tiên bộ Docker offline. Chạy `run_demo_docker.ps1` trên Windows
hoặc `bash run_demo_docker.sh` trên Linux; lần đầu tạo engine cho GPU đích.
Demo có Video/RTSP, Kiểm tra ảnh, Lịch sử và **Hiệu năng** (biểu đồ).
Trang Nghiên cứu đã gỡ khỏi demo; báo cáo vẫn có trong gói tài liệu/offline.
Xem [hướng dẫn Docker](docs/STRAW_DOCKER_DEMO.md).

**v3.4:** nguồn xử lý ảnh giữ đủ chi tiết ở cả 1–3 sản phẩm; preview được
thu nhỏ riêng, không dùng để chấm nét hay đưa vào CNN.
[Chi tiết và kiểm chứng](docs/STRAW_PHOTO_QUALITY_V3_4.md).

**v3.3:** tracking ổn định hơn với thẻ ảnh và ghép theo thời gian; kết quả cần
xem lại hiện rõ lý do. [Chi tiết sửa tracking](docs/STRAW_TRACKING_FIX_V3_3.md).

**Bản v3.2:** băng chuyền ảnh nhận tối đa **50 ảnh/lượt**, theo dõi và phân loại
từng sản phẩm khi nhiều sản phẩm cùng nằm trong ROI.

**Bản v3:** chọn ảnh lên băng chuyền, 1–3 sản phẩm đồng thời, xem mask từng bước,
CPU/RAM và bộ đếm GPU thực. [Hướng dẫn trình diễn v3](docs/STRAW_UPGRADE_V3.md).

**Bản v2:** giao diện chính đã có video/RTSP, kiểm tra ảnh, lịch sử và
dashboard nghiên cứu. Chọn **Phát lại mẫu → Bắt đầu kiểm tra** để thử video
đi kèm. Đọc [hướng dẫn v2](docs/STRAW_UPGRADE_V2.md).

Demo chính: **B1 512 px, pose/illumination v1**, engine FP16 được build trên
RTX 3050 Laptop. Mở PowerShell tại thư mục đã giải nén:

```powershell
.\setup_demo.ps1
.\verify_demo.ps1 -Runtime
.\run_b1_conveyor.ps1
```

Mở **http://127.0.0.1:8033/** để dùng video hoặc RTSP. Tab **Kiểm tra ảnh**
nhận ảnh trong `demo_samples/input/`. Mô phỏng ảnh cũ ở `/conveyor`;
khi qua ROI, worker TensorRT phân loại
`good / bent / loose / missing / pierced`. Không cần Node.js hoặc PyTorch
để chạy giao diện này.

```powershell
# Sau khi đã mở demo: kiểm tra thật upload → ROI → engine → kết quả
.\verify_demo.ps1 -Runtime -Inference

# Dừng server và container dành riêng cho bản giải nén này
.\stop_demo.ps1
```

Máy mới cần Python 3.12, Windows/WSL2, distro Ubuntu 24.04 tên
`TTTN-DeepStream`, NVIDIA GPU, Docker trong WSL, NVIDIA Container Toolkit và
image DeepStream đã ghi nhận. Gói ZIP chứa mã nguồn, checkpoint, ONNX, engine,
clip replay và ảnh mẫu; image SDK/WSL/môi trường Python được cài riêng.
Gói đầy đủ kèm wheel NumPy/Pillow để cài host offline; cài Python và tải SDK
lần đầu cần Internet. Xem [hướng dẫn đầy đủ](docs/DEEPSTREAM_HANDOVER.md).

**Engine gắn với môi trường build.** Nếu GPU/driver/image khác, preflight
sẽ dừng với hướng dẫn build lại; không sửa hash để bỏ kiểm tra. Đọc mục
“Máy nhận khác GPU hoặc SDK” trong hướng dẫn trước khi chạy.

Kiểm tra ZIP từ PowerShell: `Get-FileHash .\ten_goi.zip -Algorithm SHA256`,
đối chiếu file `.sha256` đi kèm. Sau giải nén, `verify_demo.ps1` đối chiếu
mọi tệp bàn giao với `release_manifest.json`.

Trạng thái và phần hồ sơ còn phải xác nhận: [HANDOVER_STATUS.md](HANDOVER_STATUS.md).
