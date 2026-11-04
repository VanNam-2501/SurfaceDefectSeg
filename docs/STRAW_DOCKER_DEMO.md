# Demo Docker v3.5 — 04/11/2026

## Phạm vi

Một image `straw-qc-demo:v3.5` chứa web, băng chuyền ảnh 1–3 sản phẩm,
kiểm tra chất lượng, tracking, DeepStream/NVDEC cho video/RTSP, ONNX và
worker TensorRT thường trú. Không gọi Docker từ trong container, không
mount Docker socket và không cần Python/PyTorch/WSL distro dự án trên máy nhận.

Demo có bốn mục: Video/RTSP, Kiểm tra ảnh, Lịch sử, Hiệu năng. `/research` và
API nghiên cứu trả 404; nghiên cứu và báo cáo vẫn có trong hồ sơ source/offline.

## Máy nhận

- Máy x86-64 có GPU NVIDIA tương thích DeepStream/TensorRT của image, driver
  tương thích và đủ VRAM để tạo/chạy engine FP16. Đã thử trên RTX 3050 Laptop.
- Windows: Docker Desktop dùng Linux containers và backend WSL2, có GPU-PV.
  [Hướng dẫn chính thức Docker](https://docs.docker.com/desktop/features/gpu/).
- Linux: Docker Engine và NVIDIA Container Toolkit đã cấu hình cho Docker.
  [Hướng dẫn NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Cần dung lượng cho image DeepStream giải nén (khoảng 29 GB), archive và
  dữ liệu. Không hỗ trợ macOS, GPU AMD hoặc máy chỉ có CPU cho đường TensorRT này.

Docker chứa môi trường phần mềm; GPU và driver vẫn do máy nhận cung cấp.
Launcher cấp một GPU cho một worker; máy có nhiều GPU vẫn dùng một thiết bị
cho cả decode và suy luận, để số liệu GPU thống nhất với thiết bị đang xử lý.
Engine không được dùng chung một cách mặc định giữa các GPU/phiên bản runtime.
[NVIDIA giải thích tính tương thích engine](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).

## Chuyển máy bằng bộ offline

Chép toàn bộ thư mục `artifacts/releases/docker_v3_5_2026-11-04/` sang máy nhận.
Image đã chứa ONNX và ảnh/video mẫu; không cần kéo image nền hoặc tải pip khi chạy.

Windows PowerShell, tại thư mục vừa chép:

```powershell
.\run_demo_docker.ps1
.\run_demo_docker.ps1 -Action status
.\run_demo_docker.ps1 -Action logs
.\run_demo_docker.ps1 -Action stop
```

Linux:

```bash
bash run_demo_docker.sh
bash run_demo_docker.sh status
bash run_demo_docker.sh logs
bash run_demo_docker.sh stop
```

Script kiểm SHA256 trước khi `docker load` nếu image chưa có, rồi khởi động
container `straw-qc-demo`. Mở **http://127.0.0.1:8033/**. Lần đầu có thể mất
vài phút để build engine trên GPU đích; theo dõi log tới khi web và GPU sẵn sàng.
Cache khóa theo ONNX, GPU/driver, TensorRT và precision; thay đổi hoặc file engine
sai hash sẽ tạo lại. Worker kiểm IO và warmup trước khi nhận sản phẩm.

Ví dụ Docker Engine chạy trong distro WSL đã cài sẵn:

```powershell
.\run_demo_docker.ps1 -WslDistro TTTN-DeepStream
```

Đổi cổng khi 8033 đã được dùng:

```powershell
.\run_demo_docker.ps1 -Port 8036 -Name straw-qc-demo-2 -Volume straw-qc-demo-data-2
```

Linux dùng `STRAW_PORT=8036 STRAW_CONTAINER=straw-qc-demo-2 STRAW_VOLUME=straw-qc-demo-data-2 bash run_demo_docker.sh`.

Chỉ publish cổng trên loopback `127.0.0.1`. Giữ kiểm tra Origin cho POST.
Không cần chế độ privileged. Trong Docker, RTSP `127.0.0.1` là container:
dùng IP LAN của camera hoặc hostname/IP nguồn mà container truy cập được.
Docker Desktop cung cấp `host.docker.internal` nếu nguồn phát nằm trên host.

## Dữ liệu và trang Hiệu năng

Volume `straw-qc-demo-data` chứa ảnh tải lên, lựa chọn ảnh, lịch sử SQLite,
crop kiểm định, queue và cache engine. Stop/start giữ dữ liệu; không dùng lệnh
xóa volume nếu cần giữ lịch sử. Một volume chỉ cho một phiên ứng dụng sở hữu.

Mục **Hiệu năng** mở `/performance`, gồm biểu đồ CPU ms/frame, số ảnh gửi/hoàn
tất trên GPU, RAM tiến trình và track quan sát/dự báo. Lấy mẫu thật mỗi giây,
giữ tối đa 300 mẫu trong RAM của server; đổi trang không mất dải mẫu đang có.
Bắt đầu phiên mới sẽ reset dải mẫu theo phiên mới. Không dựng dữ liệu quá khứ.
CPU không có frame mới sẽ để khoảng trống; lượt GPU không tính warmup/keepalive.
RAM biểu đồ là tiến trình web/xử lý ảnh, không cộng giả RAM của worker GPU.

## Build từ source

Git không chứa model weights, dataset, môi trường ảo, lịch sử người dùng hoặc
image Docker. Lấy `b1_rawrgb512.onnx` và `demo_samples/` từ gói bàn giao/assets.
ONNX phải đúng hash được khóa trong `docker/demo/model/manifest.json`.

```powershell
python scripts/deployment/prepare_docker_build.py
docker build -t straw-qc-demo:v3.5 -f .docker/context/docker/demo/Dockerfile .docker/context
```

Context được tạo theo danh sách cho phép, khoảng 64 MB, không quét cả dataset.
Build cần Internet cho base image/Python packages; bộ offline đã xuất không cần
Internet khi khởi động. Dockerfile khóa digest image DeepStream, NumPy và Pillow.
Runtime tự compile capture probe trong lúc build image, không tải gói lúc mở video.

## Kiểm chứng và giới hạn

Hồ sơ ở `artifacts/verification/upgrade_v3_5/`: kiểm thử source, GPU, decode,
biểu đồ, cache/restart và tính toàn vẹn gói. `docker_release.json` đi kèm ghi
image ID, hash archive và phạm vi đã kiểm tra. Không gán kết quả nghiên cứu
cho demo; ảnh/video phát lại chưa thay kiểm định trên camera độc lập.
