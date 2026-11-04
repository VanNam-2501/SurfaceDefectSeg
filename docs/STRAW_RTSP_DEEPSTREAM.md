# Nhánh thực nghiệm RTSP / DeepStream

> **Bản bàn giao 03/11/2026:** demo mặc định hiện là `b1_stream_512_pose_v1`
> tại 512 px, chạy bằng `run_b1_conveyor.ps1`. Lệnh cài/chạy/dừng và build
> engine trên máy nhận: [DEEPSTREAM_HANDOVER.md](DEEPSTREAM_HANDOVER.md).
> Phần bên dưới giữ hồ sơ thí nghiệm lịch sử; các mặc định 384/512 cũ không
> thay thế bản bàn giao hiện hành.

> **Cập nhật 21/10/2026:** Các launcher mặc định vẫn dùng bản TensorRT 384 tại `artifacts/b1_stream`. Bản 512 ở `artifacts/b1_stream_512` là nhánh thử nghiệm vì chỉ sửa 2/300 ảnh và chưa được kiểm trên camera độc lập. [So sánh và điều kiện triển khai](../straw_resume/error_analysis/B1_REALTIME_UPGRADE.md).

Nhánh Git `codex/rtsp-deepstream`; mã tại `apps/stream_demo/`, kết quả tại `artifacts/b1_stream/`. Dùng checkpoint B1 phục vụ web đã khóa; không chọn lại seed, không dùng test và không sửa cấu hình web HTTP.

## Kiến trúc thử

```text
300 ảnh validation -> H.264 replay -> RTSP server localhost (WSL)
 -> nvurisrcbin (NVDEC, GPU NVMM)
 -> nvstreammux (batch 1)
 -> nvinfer (nạp b1_rtx3050_fp16.engine, full frame)
 -> tensor metadata / argmax 5 lớp / JSONL
```

Đây là backend streaming độc lập. Bài đo headless ban đầu bỏ frame ở `fakesink`; chế độ xem mới ghi frame JPEG gần nhất và dự đoán sang `artifacts/b1_stream/preview/`, rồi mở giao diện loopback tại `http://127.0.0.1:8033/`. Viewer chỉ để quan sát; phép đo chuẩn vẫn lấy từ pipeline headless. Nguồn mô phỏng không thể thay camera thật để kiểm chứng nhòe, ánh sáng, codec camera, mạng vật lý hoặc ranh giới sản phẩm.

Máy Windows có RTX 3050 Laptop 4 GB. WSL cũ dùng Ubuntu 26.04 và chưa có Docker. Môi trường thử riêng `TTTN-DeepStream` dùng Ubuntu 24.04, ổ đĩa tại `.stream-runtime/Ubuntu24` trên ổ E. Không cài driver Linux vào WSL; dùng driver Windows và NVIDIA Container Toolkit. Dùng image chính thức `ghcr.io/nvidia/deepstream:9.2-samples-multiarch` và lưu image digest khi chạy. Container không dùng `--privileged`, không mở X11 hay âm thanh; RTSP server chỉ bind localhost WSL.

Lượt đo dùng container lab `tttn-b1-deepstream-lab` từ image gốc đã khóa và biên dịch C++ trực tiếp; container được dừng sau test, giữ thư viện để chạy lại nhanh. Dockerfile là phương án đóng gói tùy chọn, không phải image đã dùng cho phép đo này. Legacy Docker builder trên máy xử lý các bước commit rất chậm nên không dùng nó cho lượt thử cuối. Image mẫu tải khoảng 11,5 GB, dung lượng image tại máy khoảng 28,6 GB; đây là SDK và bộ ví dụ, không phải kích thước mô hình CNN hay một bản runtime đã tối giản.

## Giữ đúng mô hình và tiền xử lý

`prepare_b1_stream.py` kiểm tra SHA checkpoint, manifest validation và từng proxy ảnh 640 px. Trật tự frame và sample_id nằm trong manifest ngoài video; không chèn nhãn/ID vào đầu vào CNN.

ONNX nhận RGB float NCHW trong miền 0..255. Phép `(x/255 - mean)/std` được nhúng trong graph, vì nvinfer chỉ có một `net-scale-factor` trong khi ImageNet có std riêng từng kênh. Config nvinfer dùng RGB, scale 1, offsets 0, resize vuông 384 px. Output graph là softmax 5 lớp; argmax giữ cùng ý nghĩa. Không áp ngưỡng bỏ nhãn. Parity với PyTorch được đo trên 20 ảnh validation phân bố theo thứ tự cố định trước khi chạy TensorRT; đây là kiểm tra kỹ thuật, không phải toàn bộ đánh giá chất lượng.

Hai bước được tách rõ: **build** `PyTorch → ONNX → trtexec --fp16 → b1_rtx3050_fp16.engine`; **runtime** `DeepStream → nạp .engine → suy luận GPU`. ONNX là graph trung gian của bước export/build, không phải backend chạy camera. Config `nvinfer.txt` chỉ có `model-engine-file`, không có `onnx-file`; runtime từ chối khởi động nếu thiếu engine. Engine được build bằng TensorRT trên đúng RTX 3050, batch cố định 1, workspace 512 MiB, I/O FP32 và cho phép tính toán FP16 bên trong. Một số lớp có thể vẫn dùng FP32; chi tiết định dạng/lớp nằm trong `tensorrt_layers.json`, lệnh build và kết quả ở `tensorrt_build.log`.

`inside_build_engine.sh` chỉ chạy khi cần chuyển đổi, trước khi mở nguồn thử. `engine_cache.py` kiểm tra SHA graph/engine, checkpoint, image SDK, GPU UUID/driver và cấu hình trước khi dùng lại. Cache cũ/khác được lưu vào engine_archive/ rồi build lại ngoài đường suy luận; runtime không tự lấy ONNX để thay engine. Engine được ghi provenance sau khi chạy đủ 300 frame bình thường. Các thay đổi chỉ khoảng trắng cấu hình không làm mất cache. Các lượt DeepStream tự build engine ở giai đoạn trước được lưu riêng trong `deepstream_auto_build/`.

Video H.264 dùng CRF18, không B-frame, 30 FPS, dài 10 giây. Chất lượng đo bằng 300 ảnh khác nhau, mỗi ảnh một frame; chỉ gán nhãn khi đủ 300 frame, không trùng/mất/thay thứ tự và pipeline kết thúc EOS bình thường. RTSP dùng cùng clip, thử khoảng 8 giây theo yêu cầu người dùng. RTSP có thể bắt đầu giữa video nên chưa gán nhãn mẫu từ frame_num RTSP.

## Các phép đo

1. ONNX CPU parity: chênh lệch xác suất và số nhãn thay đổi so với PyTorch trên cùng tensor.
   Đối chứng H.264 chạy cùng 300 frame qua FFmpeg CPU decode, PIL resize và mô hình PyTorch CUDA FP16 hiện tại; không báo latency của pipe Windows/WSL là tốc độ mô hình.
2. Video file + DeepStream: Macro-F1, confusion, FN/FP/sai loại trên 300 sample_id sau H.264 và resize của DeepStream.
3. RTSP + DeepStream: số frame giải mã/dự đoán, throughput theo thời gian thực, stall/error, p50/p95/p99 sau decode đến output tensor. Bài đầu khoảng 8 giây; chưa phải stress dài.
4. Ghi GPU utilization, VRAM, nhiệt và điện mỗi giây; startup gồm tạo/nạp TensorRT engine được ghi riêng. Giữ cả first frame và thống kê sau 20 frame warm-up.

Đồng hồ bắt đầu ở GPU-buffer đã decode trước nvstreammux và kết thúc khi đọc được output tensor CPU. Nó bao gồm mux/preprocess/inference/copy output; không gồm cảm biến, encode nguồn, mạng/jitter, decode hoặc browser. Không so trực tiếp số này với HTTP round-trip hoặc browser→UI. Muốn camera→kết quả cần đồng bộ timestamp nguồn hoặc phép đo bên ngoài. Nếu PTS không ghép được thì không báo latency.

## Chạy lại

Từ PowerShell ở gốc dự án:

```powershell
.\run_b1_stream.ps1 -Prepare
```

Lần tiếp theo có thể bỏ `-Prepare` để tái dùng ONNX/video. Script build probe C++ từ mã nguồn, chạy file và RTSP rồi sinh REPORT.md. Logs được giữ nếu thất bại; không chuyển sang CPU rồi gọi đó là DeepStream GPU. Launcher chuyển đường dẫn dự án sang WSL bằng wslpath.

### Xem luồng TensorRT trực tiếp

Từ PowerShell ở gốc repo chạy:

```powershell
.\run_b1_preview.ps1
```

Launcher mở viewer ở cổng 8033, dựng clip validation lặp tối đa 45 giây, phát clip qua RTSP localhost và chạy `nvinfer` với engine `b1_rtx3050_fp16.engine`. Frame sau chuyển JPEG; nhãn/xác suất được ghép PTS khi khớp trong 15 ms, nếu không trang hiển thị kết quả suy luận hoàn tất gần nhất và ghi rõ có thể lệch frame. Có thể chọn 10–300 giây bằng `-Seconds 60`. Kết thúc luồng thì ảnh cuối vẫn hiện; pipeline và RTSP server được dừng. Đây là đường hiển thị thêm chuyển đổi JPEG, không dùng để báo latency/throughput benchmark.

Viewer chỉ bind `127.0.0.1`; trình duyệt Windows lấy frame và dự đoán theo PTS từ thư mục preview mà container DeepStream ghi qua volume. Nếu WSL báo `ACCESS_DENIED`, trang sẽ mở nhưng chưa có video; cần khôi phục quyền chạy distro `TTTN-DeepStream`, sau đó chạy lại launcher. File `preview_inference.log`, `preview.jsonl` và các cặp `preview/frame_*.jpg` / `preview/frame_*.json` giữ bằng chứng gần nhất của phiên xem.

Phần `run_b1_preview.ps1` / `deepstream_probe` ở trên là thí nghiệm RTSP replay cũ; mã vẫn được giữ để tái lập, nhưng giao diện 8033 hiện đã chuyển sang mô phỏng chọn ảnh và kích hoạt AI theo ROI. Gate camera RTSP/NVMM trước `nvinfer` chưa được triển khai; không dùng bằng chứng mô phỏng để tuyên bố đã hoàn thành phần đó.

Chọn **Thêm ảnh sản phẩm** là ảnh tự vào băng chuyền, không có nút chạy. Giới hạn 30 ảnh/lần chọn, 30 MB/ảnh, 200 MB/lượt; có thể thêm ảnh khi đang chạy. Cảnh gốc trên canvas được thu nhỏ 320×138 để gate trừ nền/lọc mask/theo dõi vật thể; chỉ khi track đã xác nhận cắt đường ROI và nằm trọn trong vùng kiểm tra mới gọi `POST /api/trigger`. Trước ROI chỉ phát hiện; sau ROI giữ nhãn theo đúng sản phẩm và không gọi lặp. Phần băng tải chuyển động được loại khỏi vùng đo, nhãn và ROI chỉ được vẽ sau phân tích pixel.

Khởi chạy bằng `.\run_b1_conveyor.ps1`. Server khởi động worker TensorRT thường trú qua `run_resident.sh`; engine/context/GPU buffers nạp một lần, warm-up ba lượt, sau đó chờ queue sự kiện. Worker dùng TensorRT native/CUDA trong container NVIDIA hiện có, không dựng RTSP/H.264 cho mỗi lượt ảnh và không chạy ONNX Runtime. Ảnh upload nằm trong `artifacts/b1_stream/image_jobs/<job-id>/`; bằng chứng sự kiện/kết quả nằm trong `artifacts/b1_stream/resident_queue/*.event.json` và `*.json`, log trong `launcher.log`. `runtime.json` chứa generation/heartbeat/product_calls để xác minh không nạp lại engine. HTTP chỉ bind localhost, không gửi ảnh ra ngoài máy. Warm-up được ghi riêng, không tính thành lượt phân loại sản phẩm.

Chi tiết triển khai và kiểm thử gate nằm tại [STRAW_INTELLIGENT_INFERENCE_GATE.md](STRAW_INTELLIGENT_INFERENCE_GATE.md). Smoke test giao diện đã xác nhận chọn 3 ảnh tự chạy, tạo 3 sự kiện ROI và 3 dự đoán, nền trống không phát sinh lượt AI thêm, cùng worker generation. Chuỗi tiền xử lý trực tiếp khác đường H.264/YUV cũ; cần đánh giá chất lượng triển khai riêng. Mô phỏng dùng ảnh gốc tương ứng làm đầu vào CNN khi track cắt ROI; không suy diễn rằng CNN đã phân loại được toàn cảnh có nhiều sản phẩm.

Thiết lập môi trường lần đầu (đã thực hiện trong lượt thử hiện tại):

```powershell
wsl --install -d Ubuntu-24.04 --name TTTN-DeepStream --location E:\Project\TTTN\.stream-runtime\Ubuntu24 --no-launch --web-download
wsl -d TTTN-DeepStream -u root -- bash /mnt/e/Project/TTTN/apps/stream_demo/setup_wsl.sh
```

Nguồn camera thật có thể đặt `STRAW_RTSP_URI` trong môi trường WSL khi chạy `run_container.sh rtsp`. Không lưu URL có mật khẩu vào Git hoặc log. Báo cáo camera phải ghi rõ nguồn thực; `run_experiment.sh` mặc định vẫn dùng replay localhost để tái lập được.

## Tài liệu gốc

- [DeepStream trên WSL](https://docs.nvidia.com/deepstream/text/DS_on_WSL2)
- [Container DeepStream](https://docs.nvidia.com/deepstream/text/DS_docker_containers)
- [nvinfer, chuẩn hóa và tensor metadata](https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvinfer)
- [nvurisrcbin và RTSP reconnect](https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvurisrcbin)
