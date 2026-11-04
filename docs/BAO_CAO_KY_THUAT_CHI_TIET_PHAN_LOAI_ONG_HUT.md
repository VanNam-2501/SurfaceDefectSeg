# Báo cáo kỹ thuật chi tiết: phân loại khuyết tật ống hút

**Ngày tổng hợp:** 23/10/2026  
**Phạm vi:** duy nhất nghiên cứu phân loại ống hút, pipeline B0–B6, tối ưu web B1, TensorRT/DeepStream và mô phỏng gate băng chuyền. 
**Căn cứ:** mã nguồn, cấu hình, report benchmark, bảng kết quả và artifact trong workspace.
## 1. Bài toán, dữ liệu và protocol

### 1.1 Dữ liệu, protocol và tiền xử lý nghiên cứu

- Dataset gồm 2.000 ảnh, 8 thương hiệu, 5 lớp cân bằng: good, bent, loose, missing, pierced.
- Audit ghi 0 ảnh lỗi đọc/không có lớp, 0 ảnh RGB trùng tuyệt đối và 76 cặp dHash gần nhau. Grouping các cặp là heuristic, chưa xác minh hoàn toàn theo phiên chụp.
- Split confirmatory: 1.401 train / 300 validation / 299 test. Manifest có sample ID, label, group ID, SHA và protocol/experiment fingerprint; không cho group đi qua nhiều split.
- Tiền xử lý dữ liệu: đọc ảnh gốc; áp EXIF orientation bằng exif transpose trước khi chuyển RGB; resize PIL bilinear; chuẩn hóa ImageNet theo channel mean [0,485; 0,456; 0,406] và std [0,229; 0,224; 0,225].
- Với mô hình nhiều view: global view resize đúng kích thước cấu hình; local crop được cắt trên ảnh gốc trước resize, bốn cửa sổ góc chồng lấn, mỗi cửa sổ 60% khung ảnh rồi resize 224×224. Tạo crop sau biến đổi hình học chung để tọa độ/crop nhất quán.
- Cache lossless lưu view và kiểm SHA/kích thước ảnh nguồn; không chấp nhận cache từ ảnh đã resize làm nguồn gốc. Augmentation được đồng bộ giữa global/local.
- 76 cặp gần trùng chưa được rà quyết định thủ công toàn bộ; nguồn/quyền sử dụng và ranh giới nhãn pierced/loose còn cần xác minh.

### 1.2 B0–B6 và từng kỹ thuật

| Mã | Đầu vào và hợp nhất | Câu hỏi |
|---|---|---|
| B0 | Global 224, ResNet18 | Baseline ảnh toàn cảnh |
| B1 | Global 512, ResNet18 | Tăng độ phân giải |
| B2 | 4 local crop 224, mean pooling | Local riêng |
| B3 | Global 224 + 4 local; mean pooling và ghép | Global bổ sung local |
| B4 | Global-local; attention score từ local feature | Attention local |
| B5 | B4 nhưng score attention nhận global context | Ngữ cảnh ảnh điều khiển trọng số vùng |
| B6 | B5 + embedding tọa độ crop | Thông tin vị trí vùng |

Mô hình dùng shared ResNet18. B4–B6 sinh score từng crop, softmax score thành trọng số attention rồi tính tổng có trọng số các local feature. B5 đưa global feature vào score; B6 thêm embedding tọa độ crop. Classifier nhận global representation ghép local aggregate. Attention là cơ chế chọn feature, không phải detector; không có mask/box ground truth để xác nhận định vị.

Train confirmatory dùng AdamW, encoder LR 1e-4, head LR 1e-3, weight decay 1e-4, tối đa 40 epoch, patience 10, AMP CUDA, effective batch 32, seed 42/43/44. B1 dùng batch 2 và accumulate 16; các cấu hình khác batch 4 và accumulate 8. BN running statistics được giữ cố định ở cấu hình global-local; affine parameters vẫn học. Checkpoint chọn theo validation macro-F1; CE là phụ khi hòa.

### 1.3 Kết quả confirmatory

| Kết quả | Giá trị |
|---|---:|
| Run hoàn tất | 21/21 |
| B1 global 512 test macro-F1 | 99,34% ± 0,00 |
| B1 test accuracy | 99,33% ± 0,00 |
| B6 test macro-F1 | 97,99% ± 0,89 |
| Paired B6 − B1 | −1,35 điểm phần trăm |
| CI 95% | [−2,63; −0,32] |

Ba seed dùng cùng 299 ảnh; không diễn giải như 897 mẫu độc lập. Tập test nội bộ đã từng tham gia lịch sử phát triển dataset/model nên không phải external holdout hoàn toàn mới.

### 1.4 Pipeline và kiểm chứng kỹ thuật

Đã tạo data/model/runner; training resume; Val-only model selection; freeze trước test; test-cache provenance; benchmark end-to-end; readiness report; Kaggle notebook và code bundle có resume. Hồ sơ ghi 22 test global-local và 15 test Kaggle-session pass. Có smoke 7 cấu hình trên subset/random-init/1 epoch, chỉ chứng minh code path hoạt động. Pilot seed 42 ba epoch cho B0 Val F1 0,9672, B5 0,9571, B1 512 khoảng 0,9808; không coi đây là kết quả xác nhận. Full notebook chưa được xác minh chạy trên Kaggle thật.

## 2. Tối ưu B1 cho web: đầu vào, runtime, hậu xử lý

### 4.1 Đường xử lý ảnh triển khai

Trong runtime PyTorch B1:

1. Nhận byte ảnh, decode bằng PIL.
2. Đọc EXIF orientation và xoay ảnh vật lý đúng hướng bằng ImageOps.exif_transpose.
3. Chuyển ảnh về RGB, bỏ phụ thuộc BGR.
4. Resize vuông bilinear về 512 hoặc 384 tùy cấu hình; không crop theo nhãn.
5. Browser camera gửi JPEG q90; offline benchmark dùng PIL JPEG làm proxy. Nội suy/encoder trình duyệt không hoàn toàn giống PIL nên đã có phép browser replay riêng.
6. Chuyển HWC uint8 thành NCHW float32, chia 255, rồi chuẩn hóa theo ImageNet mean/std.
7. Đưa tensor lên CUDA; với runtime chọn FP16 đổi dtype half; channels-last chuyển memory format.
8. Chạy batch=1; logits được chuyển về float32/CPU để softmax ổn định và argmax.
9. API trả class, probabilities/confidence, timing và frame_id; UI chỉ gắn kết quả với request tương ứng.

Không có morphology/NMS trong hậu xử lý phân loại ảnh; đó là kỹ thuật dành cho mask/box, không áp dụng cho classifier 5 lớp.

### 4.2 Các tối ưu đã thử và kết quả chất lượng

| Cấu hình | Macro-F1 | FN | FP | Sai loại lỗi | Kích thước ảnh gửi TB |
|---|---:|---:|---:|---:|---:|
| ONNX FP32 JPEG512 | 99,682% | 1 | 0 | 0 | 39,1 KiB |
| ONNX INT8 JPEG512 | 96,898% | 1 | 0 | 8 | 39,1 KiB |
| PyTorch FP16 channels-last JPEG384 | 97,739% | 4 | 0 | 3 | 20,7 KiB |
| PyTorch FP16 channels-last JPEG512 | 99,682% | 1 | 0 | 0 | 39,1 KiB |
| PyTorch FP16 JPEG512 | 99,682% | 1 | 0 | 0 | 39,1 KiB |
| PyTorch FP32 ảnh gốc | 99,682% | 1 | 0 | 0 | 3602,9 KiB |
| PyTorch FP32 JPEG512 | 99,682% | 1 | 0 | 0 | 39,1 KiB |

JPEG 512 q90 mô phỏng PIL đổi 0/300 dự đoán argmax so với ảnh gốc. JPEG giảm payload lớn nhưng không tương đương resize/capture của mọi camera.

### 4.3 Benchmark runtime

Các dòng ONNX CPU và QAT TorchScript INT8 CPU là đối chứng triển khai trên CPU: ONNX Runtime được khởi tạo với CPUExecutionProvider; QAT được xuất TorchScript INT8 và chạy bằng backend CPU oneDNN. Chúng không phải cấu hình web chính và không phải TensorRT GPU. ONNX là định dạng model, không tự quyết định thiết bị chạy; provider/runtime mới quyết định CPU hay GPU. Web chính dùng PyTorch CUDA, còn nhánh streaming dùng TensorRT FP16 trên GPU.

Batch=1, cùng 20 sample ID validation, 20 warm-up, 150 lần đo mỗi cấu hình; không train đồng thời. Tổng là decode → logits → softmax, chưa gồm browser/mạng/HTTP.

| Runtime | Preprocess p50 | Infer + transfer p50 | Tổng p50 | Tổng p95 |
|---|---:|---:|---:|---:|
| GPU FP32 | 5,64 ms | 8,14 ms | 14,02 ms | 22,08 ms |
| GPU FP16 channels-last | 5,49 ms | 8,81 ms | 14,41 ms | 16,99 ms |
| ONNX CPU FP32 | 5,33 ms | 89,31 ms | 95,09 ms | 105,94 ms |
| ONNX CPU INT8 | 5,49 ms | 95,74 ms | 101,41 ms | 117,67 ms |
| QAT TorchScript INT8 CPU | 5,31 ms | 56,12 ms | 61,58 ms | 73,33 ms |
| Fine-tuned 384 GPU FP16 channels-last | 2,70 ms | 8,47 ms | 11,27 ms | 11,70 ms |

FP16 không nhất thiết nhanh hơn FP32 ở batch nhỏ: p50 tổng trong lượt này nhỉnh hơn; p95 thấp hơn. Channels-last/precision phải benchmark trên workload thật.

### 4.3.1 Cấu hình triển khai mục tiêu: máy NVIDIA GPU

Do mục tiêu triển khai là máy có GPU NVIDIA, bảng CPU phía trên chỉ giữ vai trò đối chứng lịch sử; không dùng nó để chọn cấu hình triển khai GPU. Đã bổ sung phép đo riêng trên NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), driver 610.60, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, TensorRT 11.4.0.106. Đây là phép thử cùng checkpoint B1 web 384 seed 42 (SHA-256 `2a1881fcd9c17201ecfc6b6547b699c07e9d0ff8ba1a250ee943adf0f27cf852`), batch=1, đúng cùng 300 ảnh validation và cùng byte JPEG đầu vào cho tất cả runtime. Không dùng test để chọn runtime; không chạy train đồng thời. Mỗi cấu hình có 30 warm-up và 150 lượt đo; số liệu là thời gian p50/p95/p99.

Tiền xử lý chung: đọc ảnh gốc và kiểm SHA theo manifest → áp EXIF transpose → RGB → resize bilinear 384×384 → encode JPEG q90 → decode cùng payload → tensor NCHW float32, chia 255 và chuẩn hóa ImageNet. Phần thời gian preprocess trong bảng là decode/resize/chuẩn hóa ở máy benchmark; không cộng thời gian mạng HTTP. Với TensorRT và ONNX, tensor I/O vẫn FP32; TensorRT chạy các node được chuyển sang FP16. Đây là lựa chọn chủ ý để giữ giao diện đầu vào/đầu ra số học ổn định, còn phép tính bên trong engine tận dụng FP16.

| Runtime trên GPU | Độ chính xác phép tính | Accuracy | Macro-F1 | FN/240 lỗi | FP/60 tốt | Sai loại lỗi | Preprocess p50 | Infer + transfer p50 | Tổng p50 | Tổng p95 | Tổng p99 | Engine/model |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CUDA | FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 2,050 ms | 5,453 ms | 7,475 ms | 8,955 ms | 10,481 ms | checkpoint |
| PyTorch CUDA | FP16 | 99,667% | 99,682% | 1 | 0 | 0 | 2,053 ms | 5,525 ms | 7,514 ms | 10,306 ms | 10,783 ms | checkpoint |
| PyTorch CUDA channels-last | FP16 | 99,667% | 99,682% | 1 | 0 | 0 | 4,090 ms | 12,146 ms | 16,305 ms | 17,740 ms | 18,997 ms | checkpoint |
| ONNX Runtime CUDA EP | FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 2,554 ms | 5,612 ms | 8,386 ms | 9,048 ms | 10,008 ms | ONNX |
| ONNX Runtime CUDA EP | FP16 nội bộ, I/O FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 4,375 ms | 7,829 ms | 12,423 ms | 15,710 ms | 25,183 ms | ONNX |
| **TensorRT CUDA** | **FP16 nội bộ, I/O FP32** | **99,667%** | **99,682%** | **1** | **0** | **0** | **2,487 ms** | **1,971 ms** | **4,785 ms** | **6,709 ms** | **7,454 ms** | **engine 24.43 MB** |

Trên đúng bộ đo này, TensorRT FP16 giảm tổng latency khoảng 36,0% ở p50 và 25,1% ở p95 so với PyTorch CUDA FP32. Kết quả validation TensorRT và PyTorch FP32 khớp hoàn toàn: ma trận nhầm lẫn, accuracy, macro-F1, FN/FP và số sai loại lỗi. Có 1 lỗi bỏ sót trong tập 240 ảnh lỗi; không được hiểu là mô hình không còn FN. ONNX Runtime CUDA chạy đúng trên GPU (profile xác nhận node được thực thi bởi CUDA EP, không bởi CPU EP), nhưng ở batch=1 này không nhanh hơn TensorRT; ONNX FP16 còn chậm hơn ONNX FP32. PyTorch channels-last cũng chậm hơn trên RTX 3050 ở cấu hình/kích thước này, nên không bật chỉ vì định dạng đó thường được quảng bá cho GPU.

**Cấu hình khuyến nghị cho máy NVIDIA trong phạm vi phép đo:** checkpoint B1 web 384 → ONNX FP16 explicit graph → build TensorRT engine FP16 trên đúng GPU đích → chạy TensorRT CUDA batch=1, I/O FP32; giữ tiền xử lý 384/JPEG q90/chuẩn hóa ImageNet và hậu xử lý softmax/argmax hiện tại. Artifact engine phụ thuộc GPU, driver và phiên bản TensorRT nên cần build/kiểm tra trên máy triển khai cuối; không chép engine RTX 3050 sang một GPU khác như thể engine phổ quát. Nếu cần ưu tiên tính đơn giản hoặc khả năng dùng lại checkpoint, dùng PyTorch CUDA FP32 làm đường dự phòng; số đo hiện tại cho thấy ONNX Runtime CUDA chưa đem lại lợi thế tốc độ ở workload này.

INT8 TensorRT đã được thử theo hướng hiệu chuẩn 200 ảnh Train (40 ảnh/lớp), dựng Q/DQ cho convolution, giữ classifier nhạy ở FP32, và sửa biểu diễn bias Q/DQ để parser TensorRT đọc graph. Parser qua được nhưng TensorRT 11.4 không tạo được serialized engine trên RTX 3050; builder trả về `None` khi tìm tactic/implementation cho graph lượng tử hóa. Vì không có engine và không có inference, **không có kết quả latency/accuracy INT8 TensorRT để báo cáo**. QAT/ONNX INT8 CPU ở bảng lịch sử không thay thế được số đo TensorRT GPU. Đây là giới hạn của graph/build hiện tại, không phải bằng chứng rằng mọi cấu hình INT8 trên mọi GPU NVIDIA đều không chạy được.

Chi tiết script, pinned dependencies, cache đầu vào, manifest hiệu chuẩn, profile provider, engine build metadata và kết quả máy được lưu ở `scripts/research/benchmark_b1_nvidia_gpu.py`, `requirements/b1-nvidia-gpu-study.txt`, `requirements/b1-tensorrt-cu12.txt`, `run_b1_nvidia_gpu.ps1` và `artifacts/b1_nvidia_gpu/`.

### 4.4 Fine-tune kích thước giảm và lượng tử hóa

- Fine-tuned 384 trên ba seed: B1 trung bình macro-F1 99,571% → 99,457%, giảm khoảng 0,114 điểm. Với từng seed, số FN/FP không đổi so với checkpoint B1 cùng seed: seed42 1/1 FN và 0/0 FP; seed43 1/1; seed44 2/2.
- Artifact web được chọn cố định seed42, input384, FP16, channels-last, CUDA, autotune, JPEG q90. Tiêu chí engineering là giảm F1 tối đa 0,5 điểm, FN≤1, FP≤0 trên validation; đây không phải chứng minh non-inferiority thống kê.
- QAT CPU INT8: 5 epoch, macro-F1 96,536%, FN1 FP0; PTQ ONNX INT8 96,898%, FN1 FP0, subtype error 8. Khác runtime/backend giữa PTQ ONNX và QAT TorchScript nên không quy toàn bộ chênh lệch cho thuật toán lượng tử hóa.
- QAT TorchScript dùng CPU oneDNN; không phải TensorRT INT8 hoặc ONNX Runtime GPU.
- Temperature scaling group cross-fit không được bật vì NLL xấu đi 0,0213 → 0,3416, Brier 0,00799 → 0,00813. Giữ T=1. Nhiệt độ dương không đổi argmax; không có lý do để bật hậu xử lý chỉ vì có thể hiệu chuẩn.
- Ngưỡng defect score được xem xét trên validation. Nếu ngưỡng làm giảm bỏ sót nhưng tăng sai loại hay không giảm tổng lỗi thì không được mô tả là cải thiện tổng quát.

### 4.5 Web HTTP và trình duyệt

- Web model worker nạp checkpoint một lần, warm-up, một request đang chạy và tối đa một request chờ với ngân sách chờ 50 ms; quá tải có thể trả 429. Trả frame_id để tránh hiện kết quả cũ khi client đổi trạng thái.
- Server giữ model GPU; CPU decode/resize/chuẩn hóa và serialize JSON. Không tạo queue vô hạn; không khởi tạo random weights khi thiếu checkpoint.
- HTTP localhost khoảng 30 giây, mục tiêu 10 req/s/client:

| Client | Thành công/tổng | req/s | HTTP p50 | p95 | p99 | 429/lỗi | >200 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 297/297 | 9,89 | 32,05 ms | 33,92 ms | 34,48 ms | 0/0 | 0 |
| 2 | 593/593 | 19,70 | 33,31 ms | 39,81 ms | 40,79 ms | 0/0 | 0 |

Peak RSS server 1316,1 MiB ở phép đo ngắn; chưa chứng minh bộ nhớ ổn định 30 phút. Bản từ chối ngay khi GPU bận có 297/594 request nhận 429 ở bài hai-client trước đó; queue hữu hạn đạt 593/593 ở lượt đo sau.

Browser replay: proxy ảnh 640 → canvas resize 384 → JPEG q90 → API thật → UI. Accuracy 99,333%, macro-F1 99,340%, FN1, FP0, một subtype error. Latency p50/p95/p99: capture/decode nguồn tới UI 75,35/82,41/92,33 ms; HTTP 34,60/35,70/36,41 ms; canvas resize/JPEG 9,90/12,31/15,05 ms. Có 1/300 frame vượt 200 ms (226,5 ms), được giữ trong thống kê. Đây là ảnh replay tĩnh, không đo sensor/camera thật.

## 3. TensorRT và DeepStream: tiền xử lý, build, hậu xử lý

### 5.1 Build và runtime

- Checkpoint B1 seed42 được export PyTorch → ONNX để tạo graph → TensorRT trtexec build engine FP16 trên RTX 3050 Laptop, batch=1, workspace 512 MiB.
- Runtime DeepStream nạp file .engine trực tiếp qua nvinfer; ONNX không được dùng làm backend chạy luồng. Engine cache được khóa theo SHA checkpoint/graph, GPU UUID/driver, SDK/container và config.
- Engine cho phép TensorRT chọn FP16, một số lớp nhạy có thể giữ FP32. I/O tensor là FP32.
- DeepStream graph nhận RGB NCHW 0..255. ImageNet normalize được nhúng trong ONNX graph vì nvinfer chỉ có một net-scale-factor, không thể biểu diễn trực tiếp std khác nhau theo channel. Cấu hình nvinfer RGB, scale 1, offsets 0; resize vuông 384 hoặc 512 tùy engine.
- Output graph là softmax 5 lớp; hậu xử lý lấy argmax. Không có threshold reject, NMS hay ROI trong nhánh replay cơ bản.
- Parity ONNX CPU trên 20 ảnh: size384 max absolute probability difference 4,66e-9; size512 1,19e-7; normalize graph error 0 và prediction đổi 0/20. Đây là kiểm tra export/numerical parity nhỏ, không phải benchmark chất lượng toàn tập.
- Mã hóa replay: 300 ảnh validation thành video H.264 640×640, CRF18, 30 FPS, không B-frame; giữ manifest ngoài video để map sample_id, kiểm đủ 300 frame/thứ tự/EOS trước khi chấm chất lượng.

### 5.2 Chất lượng DeepStream

| Engine/input | Macro-F1 PyTorch cùng clip | Macro-F1 DeepStream | Accuracy DeepStream | FN | FP | Sai loại | Nhãn khác biệt |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 384 FP16 | 99,340% | 99,016% | 99,000% | 1 | 0 | 2 | 1/300 |
| B1 512 FP16 | 99,682% | 99,682% | 99,667% | 1 | 0 | 0 | 0/300 |

Cổng kỹ thuật đặt mức giảm macro-F1 tối đa 0,5 điểm so với B1 512 FP32 và không tăng FN/FP. Engine 512 đạt cổng trên replay; engine384 giảm 0,666 điểm so với mức tham chiếu 99,682% và chưa đạt cổng gốc. Engine512 chỉ sửa 2 lỗi so với nhánh384, không có test camera độc lập; quyết định trong report là **không thay engine mặc định**.

### 5.3 Throughput, latency, startup

| Nguồn | Size | Frames dự đoán | Throughput quan sát | p50 sau decode | p95 | p99 | First prediction startup |
|---|---:|---:|---:|---:|---:|---:|---:|
| File replay | 384 | 300 | — | 2,81 ms | 4,27 ms | 13,16 ms | 1,29 s |
| RTSP replay | 384 | 252 | 31,03 FPS | 2,28 ms | 11,31 ms | 16,72 ms | 1,31 s |
| File replay | 512 | 300 | — | 3,33 ms | 5,31 ms | 13,85 ms | 1,79 s |
| RTSP replay | 512 | 249 | 31,02 FPS | 3,38 ms | 15,87 ms | 20,46 ms | 1,36 s |

Sau khi bỏ 20 frame đầu, RTSP 384 p50/p95/p99 là 2,28/11,25/11,60 ms; RTSP512 là 3,38/13,29/16,03 ms. Các percentile “tất cả” bị ảnh hưởng bởi startup/outlier lớn, nên phải báo cả giai đoạn warm steady-state.

Timing bắt đầu sau decode trước nvstreammux và kết thúc khi đọc tensor output CPU: gồm mux/preprocess/inference/copy output; không gồm sensor, encoder camera, mạng/jitter, video decode hay browser. FPS replay không phải sản phẩm/giây. RTSP 384 startup ban đầu từng là 53,02 giây do engine bị dựng lại vì cache path sai; sau sửa cache, lần đo là 1,31 giây. Lượt đầu được lưu, không loại khỏi bằng chứng.

## 4. Mô phỏng băng chuyền: gate pixel, tracking và trigger

Gate trong apps/stream_demo chạy trên canvas 320×138, chỉ nhìn pixel; không nhận tên file, nhãn, vị trí sản phẩm đã biết hoặc dự đoán model.

### Chuỗi xử lý

1. Hiệu chuẩn nền trống: chuyển RGB sang grayscale theo 0,299R + 0,587G + 0,114B; lưu background và previous frame.
2. Chọn strip hàng trên làm mẫu biến thiên sáng. Tính median của sai khác sáng strip so với nền để bù lệch sáng toàn khung, ít nhạy với một cạnh/speck sáng hơn mean.
3. Trên band dọc 12%–80% ảnh, tính absolute difference giữa gray hiện tại, background và brightness offset; ngưỡng pixel mặc định >18/255.
4. Tính motion pixels từ chênh frame liên tiếp >12.
5. Morphological closing bằng kernel 3×3: dilation rồi erosion để nối lỗ/đứt nhỏ trong mask.
6. Connected components 4-neighbor, tính area, bounding box, centroid, fill ratio và tỷ lệ pixel trong ROI.
7. Lọc component nếu area<120 px, width<12, height<20 hoặc fill ratio<0,45.
8. Ghép blob với track gần nhất dựa khoảng cách centroid tới vị trí dự đoán; ngưỡng 18 px. Vận tốc làm trơn 0,5 cũ + 0,5 displacement; cho mất tối đa 1 frame.
9. Track cần xuất hiện ít nhất 4 trong cửa sổ 5 frame; vùng mask cần overlap ROI ít nhất 80%; tâm di chuyển phải→trái và cắt vạch trigger.
10. Mỗi track chỉ gọi một lần; gửi event /api/trigger, worker suy luận ảnh gốc tương ứng và kết quả gắn lại đúng sản phẩm/request. Không ghép được event thì UI báo riêng, không gán nhãn cho vật thể khác.

ROI demo khoảng x=375,y=60,w=250,h=276 trên canvas 1000×?; gate analysis dùng ROI tương ứng trên ảnh nhỏ. Băng chuyền canvas và sản phẩm dịch chuyển là mô phỏng, không phải camera feed thật.

### Kết quả, giới hạn

Smoke UI: 3 ảnh được chọn tự chạy, tạo 3 ROI event và 3 dự đoán; cảnh nền trống không kích hoạt thêm AI. Worker TensorRT thường trú nạp engine/context/GPU buffers một lần và warm-up ba lượt. Đây là kiểm tra tích hợp kỹ thuật.

Chưa có video gán product ID, thời điểm vào/ra ROI và frame mục tiêu; vì vậy chưa đo gate event recall, miss-per-product, false trigger/phút, latency theo sản phẩm, hoặc so gate+B1 với B1 mọi frame. Cổng fail-open/UNCERTAIN và kiến trúc trigger-camera thật nằm ở thiết kế nghiên cứu; chưa chứng minh không bỏ sản phẩm. Chưa chạy camera IP/PLC, motion blur/ánh sáng thực hoặc stress 30 phút.

## 5. Nhánh cascade/review lịch sử

Đây là pipeline đời trước, không thuộc kết quả B0–B6 confirmatory.

| Hệ thống | Tập | FNR | FPR | Review rate | Ghi chú |
|---|---|---:|---:|---:|---|
| Full5 | Clean | 5,00% | 30,00% | 0% | Tự động phân loại |
| Cascade | Clean | 0% | 0% | 83% | Phần lớn chuyển người review |
| Full5 | Degraded v2 | 4,58% | 48,33% | 0% | Sai nhiều hơn trên suy giảm |
| Cascade | Degraded v2 | 0% | 0% | 97% | Gần như toàn bộ review |

Cascade không thể được xem là quyết định tự động hiệu quả chỉ vì FNR bằng 0; mức review quá cao. Bộ degraded đời đầu bị lỗi ghi đè tên file giữa thương hiệu, đã tạo lại v2 với filename/hash duy nhất.

## 6. Hạn chế và công việc tiếp theo

1. Xác minh định nghĩa lớp pierced/loose, nguồn/quyền dataset, và rà thủ công 76 cặp gần trùng.
2. Hoàn thành phân tích B0–B6 theo từng RQ, lỗi theo lớp và chi phí; đối chiếu toàn văn related work.
3. Hoàn tất full Kaggle run nếu cần bằng chứng môi trường đích; kiểm notebook hiện mới có kiểm thử bundle/synthetic.
4. Thu video thực có product ID/ROI event; test camera, ánh sáng, tốc độ, blur và tải kéo dài trước khi tuyên bố gate cho sản xuất.
5. Giữ kết quả 384 web, 384 DeepStream, 512 DeepStream và B0–B6 thành các dòng thí nghiệm riêng; checkpoint/decoder/resize/mục tiêu benchmark khác nhau.
6. Điền nhật ký phát triển/thực tập chỉ theo dữ liệu thực tế; hoàn thiện hồ sơ AI/prompt archive nếu yêu cầu nộp.
7. Đối chiếu yêu cầu trường/giảng viên trước khi kết luận hồ sơ đã đủ nộp.

## 7. Kết luận kỹ thuật

Đã xây dựng quy trình nghiên cứu tái lập cho hai bài toán, các mô hình và decision policy, cùng ứng dụng review/web và prototype GPU streaming. Trên nghiên cứu ống hút, B1 512 là baseline tốt nhất trong thí nghiệm confirmatory; B6 không cải thiện. Với triển khai, JPEG512 giảm payload từ khoảng 3,6 MiB xuống 39,1 KiB mà không đổi argmax trong mô phỏng PIL; fine-tune384 giúp giảm thời gian web và giữ chất lượng validation gần baseline theo tiêu chí engineering đã đặt. DeepStream TensorRT512 tái hiện chính xác dự đoán PyTorch trên 300 ảnh replay; DeepStream384 đổi 1/300 nhãn. Các phép đo này chứng minh tích hợp kỹ thuật trên phần cứng/replay đã thử, chưa chứng minh vận hành công nghiệp.

### Hồ sơ bằng chứng chính

- Nhánh ống hút: docs/STRAW_RESEARCH_PLAN.md, docs/STRAW_GLOBAL_LOCAL_VERIFICATION.md, docs/STRAW_VERIFICATION.md.
- Web optimization: artifacts/b1_deployment/REPORT.md, straw_resume/error_analysis/B1_REALTIME_UPGRADE.md.
- DeepStream: artifacts/b1_stream/REPORT.md, artifacts/b1_stream_512/REPORT.md, docs/STRAW_RTSP_DEEPSTREAM.md.
- Gate: apps/stream_demo/conveyor_gate.js, docs/STRAW_INTELLIGENT_INFERENCE_GATE.md.
- Tổng quan: docs/BAO_CAO_TONG_QUAN_KHOA_LUAN_ONG_HUT.md, docs/BAO_CAO_TONG_QUAN_TIEN_DO_DU_AN_ONG_HUT.md.
