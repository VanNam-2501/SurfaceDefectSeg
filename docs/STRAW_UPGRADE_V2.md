# Bản nâng cấp v2 — video, chất lượng ảnh, lịch sử và nghiên cứu

## Các màn hình

| Đường dẫn | Chức năng |
|---|---|
| `/` hoặc `/video` | Phát lại mẫu, tải video hoặc nhập RTSP; ROI, chiều di chuyển, hiệu chuẩn nền |
| `/images` | Kiểm tra ảnh đơn/lượt ảnh, kiểm tra chất lượng và lưu kết quả |
| `/history` | Lịch sử bền vững, lọc theo phiên/kết quả/lớp/mã, ảnh và xuất CSV |
| `/research` | B0–B6, đường học Validation, pose/illumination và thư viện hồ sơ gốc |
| `/conveyor` | Mô phỏng ảnh cũ; vẫn dùng cùng engine và ghi lịch sử |

Giao diện hỗ trợ desktop và màn hình hẹp, trạng thái đang xử lý/lỗi/kết nối lại,
ảnh thumbnail trong lịch sử và chi tiết truy nguyên. Màn hình nghiên cứu hiển thị
mean/SD theo seed; kết quả triển khai và pipeline lịch sử nằm ở các nhóm riêng.

## Luồng video

```text
Video / RTSP
  → nvurisrcbin (NVDEC, bắt buộc GPU NVMM)
  → nvvideoconvert + preview tối đa 30 frame/s + JPEG
  → luồng MJPEG liên tục cho trình duyệt
  → gate/tracking CPU riêng, tối đa 10 lượt/s trên ảnh nhỏ 320×180
  → chọn frame trong ROI
  → kiểm tra chất lượng
  → worker TensorRT thường trú hoặc bản ghi CẦN XEM LẠI
  → SQLite và giao diện
```

Giữ tỷ lệ khung hình khi thu nhỏ, tối đa 960×720. Việc chuyển JPEG sang host
là một phần của đường quan sát; không gọi toàn bộ pipeline này là zero-copy.
Engine/checkpoint vẫn là `b1_stream_512_pose_v1`; không huấn luyện/chọn lại model.

Tracking dùng vùng tiền cảnh và ghép tâm có dự báo vận tốc, xác nhận 4/5 frame,
ít nhất 80% mask trong ROI và cắt vạch đúng chiều. Một track được ghi nhận một lần.
Sản phẩm vào ROI nhưng mất track trước khi xác nhận được lưu thành trường hợp
cần xem lại nếu có ảnh ứng viên. Đây là tracker cho camera cố định, một làn,
sản phẩm tách nhau; không phải NvDCF, detector được huấn luyện hay bằng chứng
đã giải quyết các trường hợp che khuất/nhiều sản phẩm dính nhau.

Đầu nguồn cần nền trống khoảng một giây. Có thể vẽ ROI trên khung hình trước
khi bắt đầu hoặc nhập tọa độ phần trăm. Hiệu chuẩn lại khi không có track.
Ngưỡng gate và chất lượng được khóa trong metadata của từng phiên.

## Chọn frame và kiểm tra chất lượng

Trong các frame trước lúc cắt ROI, chọn ưu tiên ảnh đủ điều kiện rồi đến điểm
độ nét/tương phản trong cửa sổ 0,8 giây gần nhất. Lưu PTS của frame được chọn
và PTS sự kiện riêng biệt. Ảnh kiểm định gắn đúng frame/PTS và track trong lịch sử.
Overlay preview ước lượng vị trí giữa các lượt tracking, không dùng để chứng
minh nhãn chính xác ở từng frame video.

Preview và phân tích dùng hai luồng riêng; phân tích chỉ giữ frame mới nhất
để tránh hàng đợi cũ. Trình duyệt nhận MJPEG thay cho tải lại ảnh mỗi 400 ms.
Mô phỏng ảnh cũng chuyển gate sang Web Worker, đồng bộ chuyển động sản phẩm
và vân băng tải theo cùng quãng đường, dừng khi tab ẩn và tiếp tục theo đồng hồ
thực. Tốc độ mô phỏng vẫn là px/s; chưa có tham chiếu kích thước để quy đổi m/s.

`scripts/deployment/benchmark_preview.py` đo nhịp nhận JPEG từ HTTP, tách với
FPS inference và tần số quét màn hình. Hồ sơ `preview_cadence.json` ghi số đo
thực tế; không lấy tốc độ đường truyền này làm latency cảm biến–kết quả.

Kiểm tra chất lượng trên vùng vật thể, bỏ phần nền đệm phục vụ CNN; tính trên
ảnh xám chuẩn hóa kích thước 256×256 và bỏ viền. Các ngưỡng mặc định là tiêu chí
kỹ thuật ban đầu: độ sáng 30–240, độ tương phản tối thiểu 10, phương sai Laplacian
tối thiểu 18, cạnh tối thiểu 48 px. Người dùng có thể chỉnh độ sáng/độ nét cho
phiên mới; cần hiệu chuẩn các ngưỡng bằng video Validation của camera thực.

Ảnh quá mờ/tối/sáng/nhỏ hoặc vật thể bị cắt biên được ghi `review`, không gọi
engine rồi coi là Good. `review` là trạng thái kiểm định, không phải lớp CNN thứ sáu.
Lỗi nguồn/GPU là `error`, tách với kết quả phát hiện sản phẩm lỗi. Điểm softmax
chưa hiệu chuẩn được ghi rõ trong chi tiết; không bảo đảm ảnh ngoài miền dữ liệu
luôn bị chặn bởi các phép đo chất lượng này.

## Lưu trữ và bảo toàn danh tính

`artifacts/inspection/history.sqlite3` lưu phiên và từng sản phẩm. Ảnh tại
`artifacts/inspection/photos/`, `sessions/` và các lượt `image_jobs/`. Ghi lại
ID track, ảnh được chọn, PTS, chất lượng, lớp/xác suất, engine/checkpoint/contract
hash và thời gian. Khóa sự kiện giúp chống ghi lặp; kiểm tra generation/hash
tiếp tục được thực hiện bởi worker. Dữ liệu còn sau khi khởi động lại server.

CSV giữ bộ lọc hiện tại, có nguồn/phiên/track và hash mô hình; các chuỗi có thể
được bảng tính hiểu như công thức được vô hiệu hóa. RTSP nhận qua stdin, không
đặt thông tin đăng nhập trong dòng lệnh. Lịch sử bỏ user/password và query của
URL; không lưu URL gốc. Không đưa lịch sử, video tải lên hoặc ảnh vận hành vào
gói phát hành.

## Dashboard và hồ sơ bàn giao

`/research` đọc hồ sơ gốc, không ghi lại bảng Test hoặc freeze ledger.
`artifacts/research_dashboard/index.html` là bản offline tự chứa giao diện/dữ liệu;
các liên kết hồ sơ trỏ tương đối về tệp đi kèm. Tạo lại bằng:

```powershell
.\.venv-demo\Scripts\python.exe scripts/reporting/build_straw_dashboard.py
```

Gói v2 bổ sung báo cáo so sánh triển khai, final_results/summary, phân tích ca
sai và ảnh minh họa, bảng/figure pipeline ống hút cũ (đánh dấu lịch sử), hồ sơ
GPU/stream và báo cáo Word/PDF. Dataset đầy đủ, checkpoint của mọi run, cache
trung gian và ổ WSL/image SDK vẫn được loại khỏi gói. Danh mục thực tế và SHA
từng tệp nằm trong `release_manifest.json`.

## Kiểm chứng

```powershell
.\run_demo.ps1
# Video mẫu đã kèm gói: 5 sản phẩm; ảnh rõ, mờ và tối
.\.venv-demo\Scripts\python.exe scripts/deployment/verify_upgrade.py
# Kiểm tra đường tải video
.\.venv-demo\Scripts\python.exe scripts/deployment/verify_upgrade.py --upload --output artifacts/verification/upgrade_v2/video_upload.json
```

RTSP phát lại trong WSL:

```powershell
$videoLinuxRoot = (wsl -d TTTN-DeepStream -u root -- wslpath -u ($PWD.Path -replace '\\','/')).Trim()
wsl -d TTTN-DeepStream -u root -- bash "$videoLinuxRoot/apps/stream_demo/start_replay_rtsp.sh"
.\.venv-demo\Scripts\python.exe scripts/deployment/verify_upgrade.py --rtsp rtsp://127.0.0.1:8555/straw --output artifacts/verification/upgrade_v2/video_rtsp.json
wsl -d TTTN-DeepStream -u root -- bash "$videoLinuxRoot/apps/stream_demo/stop_replay_rtsp.sh"
```

Các lượt video tệp, video tải lên và RTSP replay được kiểm chức năng: 5 track,
3 lượt GPU, 2 review chất lượng, không đổi worker generation, ảnh chọn không
muộn hơn sự kiện và kết quả có đúng contract. Video được tổng hợp từ ảnh
Validation; không dùng số đếm này để báo accuracy/recall của camera thật.
Chưa có camera vật lý trong đợt này; các giới hạn đó vẫn giữ nguyên.

Tham khảo API: [nvurisrcbin](https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvurisrcbin),
[nvvideoconvert](https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvvideoconvert),
[GStreamer appsink](https://gstreamer.freedesktop.org/documentation/app/appsink.html).
