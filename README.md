# Phân loại khuyết tật ống hút trên hộp đồ uống

**v3.5:** demo Docker độc lập, tự build engine trên GPU máy nhận; đã bỏ trang
Nghiên cứu khỏi demo. CPU/GPU/RAM/tracking chuyển sang trang **Hiệu năng**
với biểu đồ thời gian thực. [Chạy Docker và chuyển máy](docs/STRAW_DOCKER_DEMO.md).

**v3.4:** tách nguồn ảnh 1920×1080 khỏi preview 960×540, tránh mất chi tiết
khi chạy ba sản phẩm. Giữ nguyên ngưỡng chất lượng và tốc độ hiển thị.
[Sửa cảnh báo độ nét](docs/STRAW_PHOTO_QUALITY_V3_4.md).

**v3.3:** sửa tracking khi ảnh có nền gần màu băng chuyền, giữ ID theo PTS và
hiển thị rõ lý do không gọi GPU. [Kiểm chứng 39 ảnh](docs/STRAW_TRACKING_FIX_V3_3.md).

**Nâng cấp v3:** băng chuyền từ ảnh chọn, 1–3 sản phẩm cùng lúc, sáu ảnh xử lý
trung gian, CPU/RAM và số lần gọi GPU theo phiên. [Hướng dẫn v3](docs/STRAW_UPGRADE_V3.md).

Dự án nghiên cứu CNN phân loại năm trạng thái **Good, Bent, Loose, Missing,
Pierced**, kèm demo băng chuyền dùng **TensorRT engine trong môi trường
NVIDIA DeepStream/WSL2**. Bản demo mặc định là `b1_stream_512_pose_v1`, 512 px,
worker GPU thường trú, kích hoạt khi ảnh qua ROI.

**Nâng cấp v2:** video tệp/RTSP giải mã GPU, tracking và chọn frame, kiểm tra
ảnh mờ/tối, lịch sử SQLite có xuất CSV, giao diện thống nhất và dashboard
thí nghiệm có bản offline. [Hướng dẫn và giới hạn v2](docs/STRAW_UPGRADE_V2.md).

## Chạy demo

Với bộ Docker offline, đặt image và hai script chạy trong cùng thư mục:

```powershell
.\run_demo_docker.ps1
# Docker Desktop (Linux containers / WSL2), GPU NVIDIA và driver tương thích
```

Linux: `bash run_demo_docker.sh`. Mở `http://127.0.0.1:8033/`.
Container chứa giao diện, xử lý ảnh/video, ONNX và worker TensorRT; máy nhận
không cần Python, PyTorch hoặc source. Lần đầu tự tạo engine; các lần sau
dùng cache trong volume. Xem yêu cầu GPU/runtime và hướng dẫn build lại trong
[hướng dẫn Docker](docs/STRAW_DOCKER_DEMO.md).

Đường Windows/WSL dành cho phát triển vẫn có sẵn:

```powershell
.\setup_demo.ps1
.\verify_demo.ps1 -Runtime
.\run_b1_conveyor.ps1
```

Mở **http://127.0.0.1:8033/**, chọn nguồn video hoặc **Phát lại mẫu**. Tab
**Kiểm tra ảnh** nhận ảnh riêng; mô phỏng cũ tại `/conveyor`. ZIP đầy đủ có ảnh thử trong
`demo_samples/input/`. Windows viewer chỉ cần Python 3.12 + NumPy/Pillow;
GPU inference chạy trong container NVIDIA của distro `TTTN-DeepStream`.

```powershell
.\verify_demo.ps1 -Runtime -Inference  # Kiểm tra API và engine bằng ảnh kèm gói
.\stop_demo.ps1                      # Dừng đúng server/container của thư mục này
.\package_project.ps1                # Tạo lại gói đầy đủ và gói source
```

Đọc [START_HERE.md](START_HERE.md) và
[hướng dẫn cài/runtime/build engine trên máy nhận](docs/DEEPSTREAM_HANDOVER.md).
Engine được kiểm theo GPU/driver/image build; máy khác cần build và kiểm chứng
lại từ ONNX đi kèm. Luồng canvas dùng TensorRT native; RTSP replay dùng
DeepStream `nvinfer`. Không dùng ONNX Runtime để thay đường suy luận GPU.

## Gói bàn giao

Đầu ra v3.5 tại `artifacts/releases/v3_5/`; bộ Docker offline nằm dưới
`artifacts/releases/docker_v3_5_2026-11-04/` (giữ nguyên các bản trước):

- `straw_deepstream_v3_5_YYYY-MM-DD.zip`: source ống hút, checkpoint, ONNX, engine,
  clip replay, ảnh mẫu, báo cáo Word/PDF và bằng chứng chọn lọc.
- `straw_source_v3_5_YYYY-MM-DD.zip`: source/config/tài liệu/notebook, không có
  checkpoint, ảnh, clip hoặc báo cáo nhị phân.
- `.sha256` và `release_index.json`: hash toàn gói; `release_manifest.json`
  trong mỗi ZIP ghi kích thước/hash từng tệp. `SOURCE_STATE.json` ghi commit
  nền và trạng thái working tree tại lúc đóng gói.

Gói không mang theo dataset đầy đủ, credential, môi trường ảo hoặc ổ đĩa WSL.
Lệnh đóng gói dùng trong kho phát triển có đủ artifact và dữ liệu ảnh mẫu.
Notebook trong ZIP được xóa output/execution count; notebook gốc không đổi.
Phần hồ sơ cần xác nhận được ghi tại [HANDOVER_STATUS.md](HANDOVER_STATUS.md).

## Nghiên cứu và kết quả

B0–B6 so sánh độ phân giải toàn ảnh, local views, pooling, attention, ngữ cảnh
và tọa độ với ba seed. Hồ sơ xác nhận tại
`straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1/` có 21 run,
split Train/Val/Test 1.401/300/299. Bảng nghiên cứu ghi B1 macro-F1 trung bình
99,34%; B6 thấp hơn B1 khoảng 1,35 điểm phần trăm. Đọc phạm vi và giới hạn trong
[báo cáo tổng quan](docs/BAO_CAO_TONG_QUAN_TIEN_DO_DU_AN_ONG_HUT.md).
Checkpoint pose/illumination của demo khác checkpoint B1 nghiên cứu; không
gán điểm nghiên cứu cho engine demo.

Các trường reviewer/notes trong freeze ledger còn là mẫu; cần biên bản rà
nhãn/nhóm/quyền dữ liệu thực. Demo ảnh tĩnh và replay không chứng minh đã
kiểm chứng camera nhà máy độc lập.

- [Kế hoạch nghiên cứu](docs/STRAW_RESEARCH_PLAN.md)
- [Hướng dẫn global-local](docs/STRAW_GLOBAL_LOCAL_RUNBOOK.md)
- [Điều kiện đủ bằng chứng](docs/STRAW_COMPLETION_REQUIREMENTS.md)
- [Kaggle và resume](docs/KAGGLE_ALL_IN_ONE.md)
- [Luồng DeepStream/RTSP lịch sử](docs/STRAW_RTSP_DEEPSTREAM.md)
- [Gate kích hoạt inference](docs/STRAW_INTELLIGENT_INFERENCE_GATE.md)

## Cấu trúc

```text
apps/stream_demo/       viewer, gate pixel, worker TensorRT, probe DeepStream
src/straw_global_local/ dữ liệu, B0–B6, train/evaluate/deployment
src/straw_defect_cls/   pipeline và công cụ phân loại ống hút trước đây
src/inspection_research/ công cụ nghiên cứu bổ trợ
scripts/deployment/     preflight, smoke và xác minh gói
scripts/packaging/      đóng gói theo danh mục cho phép
scripts/research/       thí nghiệm, export, Kaggle/resume
configs/               cấu hình/checkpoint hashes
tests/                 kiểm thử model, protocol, stream contract, gate
docs/                  hướng dẫn và báo cáo
artifacts/             model/kết quả/gói sinh ra; không đưa vào Git
```

Nhánh `straw-defect-classification` chỉ theo dõi đề tài ống hút. Dự án nhôm
3CAD-ANI nằm trên [nhánh main](https://github.com/VanNam-2501/SurfaceDefectSeg/tree/main).
`apps/web_demo/backend/` giữ hai API ống hút trước đây phục vụ đối chiếu;
demo hiện tại chạy từ `apps/stream_demo/`. Phạm vi file được đưa lên Git và
các đầu ra bàn giao riêng nằm trong [quy tắc mã nguồn](docs/SOURCE_SCOPE.md).

Để cài thư viện nghiên cứu từ source, dùng `python -m pip install -e ".[research,dev]"`
trong môi trường riêng với cặp torch/torchvision phù hợp thiết bị. Các kiểm thử
mặc định theo đề tài ống hút; chạy bằng `python -m pytest` hoặc `./verify.ps1`.
