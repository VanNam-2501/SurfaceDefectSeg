# Cài đặt và bàn giao demo ống hút DeepStream/TensorRT

**Bản v3** bổ sung băng chuyền từ ảnh chọn, 1–3 vật thể đồng thời, các mask
thuật toán và CPU/RAM/GPU theo phiên. [Hướng dẫn v3](STRAW_UPGRADE_V3.md).

**Bản v2** mở không gian video/RTSP, ảnh, lịch sử và nghiên cứu tại cổng 8033.
Mô phỏng ảnh trước đây được giữ ở `/conveyor`.
[Hướng dẫn v2 và kiểm chứng replay](STRAW_UPGRADE_V2.md).

## Bản demo chuẩn

| Thành phần | Bản bàn giao |
|---|---|
| Artifact | `artifacts/b1_stream_512_pose_v1/` |
| Đầu vào | RGB, NCHW float32 miền 0–255, 512×512, batch 1 |
| Chuẩn hóa | ImageNet được nhúng trong graph ONNX/engine |
| Lớp | `good`, `bent`, `loose`, `missing`, `pierced` |
| Checkpoint | `artifacts/local_b1_check/B1_pose_illum_v1/best.pt` |
| SHA-256 checkpoint | `8bed6f0944af36a616f663da534c0007514bb16f1e0442dc61bc550d3a4484e6` |
| SHA-256 engine gốc | `47bcee9abef947b492dc1548e101332739c0473db4178fc73f16dd2eb7352e09` |
| Cấu hình | `configs/straw_realtime_stream512_pose_v1.json` |
| NVIDIA SDK image | `ghcr.io/nvidia/deepstream:9.2-samples-multiarch` |
| Image ID đã kiểm | `sha256:5ee7458d5e795592e0cb182457a7f8281f2c1da7bb38ce4b8506ac693f266746` |
| Máy build | RTX 3050 Laptop, driver 610.60; UUID trong `engine_cache.json` |

Chọn ảnh → canvas băng chuyền → gate pixel xác nhận track qua ROI → HTTP
loopback → queue → worker thường trú trong container NVIDIA → serialized
TensorRT engine → kết quả gắn đúng ảnh/track. Worker giữ engine/context/buffer
trên GPU; không nạp lại mỗi sản phẩm. Windows chỉ cần NumPy/Pillow.

Giao diện băng chuyền dùng worker TensorRT native trong môi trường DeepStream.
Luồng RTSP replay dùng `nvurisrcbin → nvstreammux → nvinfer → tensor metadata`,
với mã C++ tại `apps/stream_demo/deepstream_probe.cpp`. Đây là hai đường chạy
được mô tả riêng. Demo canvas chưa triển khai gate trên video camera RTSP thật.
ONNX đi kèm phục vụ build/đối chiếu; đường suy luận demo nạp `.engine`.

## Chạy trên máy hiện tại

Giải nén vào thư mục ngắn, ví dụ `E:\StrawDemo`; mở PowerShell tại đó.

```powershell
.\setup_demo.ps1
.\verify_demo.ps1 -Runtime
.\run_b1_conveyor.ps1
```

Nếu Python chưa được tìm thấy, chỉ rõ Python 3.12:

```powershell
.\setup_demo.ps1 -Python 'C:\duong-dan\python.exe'
```

Lệnh setup tạo `.venv-demo` riêng và cài đúng phiên bản trong
`requirements/deepstream-demo.txt`. Không sao chép môi trường ảo từ máy khác.
Bản đầy đủ kèm hai wheel Windows CPython 3.12 x64 tại `runtime/wheels/demo/`;
setup ưu tiên cài offline từ đó. Python và image SDK/WSL vẫn được cài riêng.
Nếu PowerShell chặn script, mở một phiên dành cho demo:
`powershell.exe -ExecutionPolicy Bypass -NoProfile`, rồi chạy các lệnh trên.

Chọn `demo_samples/input/` để thử; ảnh đã có nhãn tham khảo trong
`demo_samples/manifest.json`. Smoke API kiểm tra liên kết ảnh/model/track,
chặn trước ROI, chống gọi lặp và cùng worker generation:

```powershell
.\verify_demo.ps1 -Runtime -Inference
.\stop_demo.ps1
```

Tham số `-Distro` hỗ trợ distro khác tên; `-Port 8035` hỗ trợ cổng khác.
Dùng cùng tham số khi chạy, kiểm tra và dừng. `-Restart` khởi động lại viewer
của đúng thư mục hiện tại. Container được đặt tên theo artifact và hash đường
dẫn, nên các bản giải nén không dùng nhầm volume của nhau.

## Cài WSL/runtime lần đầu

Chỉ thực hiện khi máy chưa có distro demo. Từ PowerShell có quyền cài WSL:

```powershell
wsl --install -d Ubuntu-24.04 --name TTTN-DeepStream --no-launch --web-download
```

Mở distro một lần để hoàn thành thiết lập nếu Windows yêu cầu. Sau đó, ở gốc
gói bàn giao:

```powershell
$demoLinuxRoot = (wsl -d TTTN-DeepStream -u root -- wslpath -u ($PWD.Path -replace '\\','/')).Trim()
wsl -d TTTN-DeepStream -u root -- bash "$demoLinuxRoot/apps/stream_demo/setup_wsl.sh"
wsl -d TTTN-DeepStream -u root -- docker pull ghcr.io/nvidia/deepstream:9.2-samples-multiarch
.\verify_demo.ps1 -Runtime
```

Máy phải có driver NVIDIA Windows hỗ trợ WSL. Script cài Docker và NVIDIA
Container Toolkit trong distro riêng. Image SDK mẫu rất lớn: hồ sơ cũ ghi
khoảng 11,5 GB tải xuống / 28,6 GB image; dự trù thêm dung lượng build và WSL.
Gói này không chứa image SDK hoặc ổ đĩa WSL. Các con số dung lượng là số đo
lịch sử trên máy phát triển, không phải kích thước tải cố định.

## Máy nhận khác GPU hoặc SDK

TensorRT có ràng buộc tương thích phần cứng/phiên bản; bản runtime này còn
kiểm tra đúng GPU UUID, driver và image ID để giữ provenance. Xem
[NVIDIA Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).
Khi preflight báo khác môi trường:

```powershell
# Dừng demo trước khi chuyển đổi
.\stop_demo.ps1
# Build lại từ ONNX sẵn có, giữ engine cũ trong engine_archive/
.\rebuild_demo_engine.ps1
# Chấp nhận hai tệp engine/cache đã được tạo lại; vẫn kiểm tra mọi tệp khác
.\verify_demo.ps1 -Runtime -TargetRebuiltEngine
.\run_b1_conveyor.ps1
.\verify_demo.ps1 -Runtime -Inference -TargetRebuiltEngine
```

Build chạy trên GPU máy nhận, biên dịch probe C++, kiểm tra đủ 300 frame file
và replay RTSP ngắn. Chỉ ghi cache provenance sau khi file replay đủ frame,
EOS bình thường và tensor hợp lệ. Không thay checkpoint, nhãn, split hoặc
chọn lại model. Đây là chuyển đổi triển khai; không tự chứng minh accuracy
camera thật. Không chỉnh `engine_cache.json` bằng tay.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Wsl/E_ACCESSDENIED` | Chạy từ PowerShell có quyền truy cập WSL; kiểm tra `wsl -l -v`. |
| Không tìm thấy Python / đường dẫn venv cũ | Cài Python 3.12, chạy `setup_demo.ps1 -Python ...`. |
| Cổng đang có dịch vụ khác | Dùng `-Port 8035` ở cả lệnh chạy/kiểm tra/dừng. |
| GPU/driver/SDK không khớp | Thực hiện quy trình build lại ở trên. |
| Container cũ mount sai đường dẫn | Bản mới có tên theo đường dẫn; không tái dùng container của checkout khác. |
| Viewer mở nhưng GPU chưa ready | Xem `artifacts/b1_stream_512_pose_v1/resident_queue/launcher.log`. |
| Ảnh ở sai chiều | Kiểm tra EXIF và tùy chọn xoay; không sửa nhãn để khớp dự đoán. |
| Muốn benchmark | Báo phạm vi timing; không dùng tốc độ canvas làm FPS camera thật. |

## Hồ sơ đi kèm

`release_manifest.json` ghi SHA-256 từng tệp; `.sha256` ngoài ZIP ghi hash
toàn gói. `SOURCE_STATE.json` ghi commit nền và trạng thái working tree lúc
đóng gói, gồm các thay đổi chưa commit. Gói không chứa `.git`, credential,
dataset đầy đủ, cache, môi trường ảo hoặc nhánh phân đoạn nhôm cũ.

Ảnh mẫu và clip là trích xuất validation dùng nội bộ cho demo, không phải
tập kiểm chứng mới. Hồ sơ nguồn/quyền dữ liệu còn chờ xác nhận ở
[HANDOVER_STATUS.md](../HANDOVER_STATUS.md). Kết quả B0–B6 dùng checkpoint
nghiên cứu, khác checkpoint pose/illumination của demo. Không gán điểm B1
nghiên cứu cho engine này.

Hướng dẫn môi trường NVIDIA: [DeepStream trên WSL](https://docs.nvidia.com/deepstream/text/DS_on_WSL2)
và [FAQ WSL](https://docs.nvidia.com/deepstream/text/DS_WSL2_FAQ).
