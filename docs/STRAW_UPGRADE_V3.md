# Straw QC v3 — xử lý ảnh, tài nguyên và nhiều sản phẩm

**v3.4:** sửa nguồn ảnh gây cảnh báo nét sai khi chạy nhiều sản phẩm; nguồn
1920×1080 được xử lý trực tiếp, preview 960×540 được tạo riêng. Xem
[kiểm chứng chất lượng](STRAW_PHOTO_QUALITY_V3_4.md).

**v3.3:** sửa tracking theo PTS, ghép Hungarian, mask RGB và định dạng thẻ ảnh.
[Nguyên nhân, cách sửa và kiểm chứng trên 39 ảnh](STRAW_TRACKING_FIX_V3_3.md).

Bản v3.2 hỗ trợ 50 ảnh/lượt; 1–3 sản phẩm cùng nằm trong ROI, mỗi sản phẩm
có track và kết quả phân loại riêng.

## Trình diễn

Mở `http://127.0.0.1:8033/`, bấm F5, chọn **Băng chuyền ảnh** và chọn tối đa
50 ảnh JPG/PNG/WebP/BMP từ máy (30 MB/ảnh). Có thể thêm tiếp, bỏ từng ảnh
hoặc bấm **Dùng 5 ảnh mẫu**. Chọn 1–3 sản phẩm đồng thời, tốc độ 120–600 px/s
và 1–3 vòng rồi bấm **Bắt đầu kiểm tra**. Các sản phẩm giữ vị trí riêng trên
băng chuyền, có thể cùng đi qua ROI trong một frame.

Danh sách ảnh đã chọn được lưu cục bộ để khôi phục khi nạp lại giao diện.

Băng chuyền ảnh được dựng trên CPU, tối đa 30 FPS. Gate nhận JPEG/pixel và PTS;
không nhận tọa độ dựng ảnh, tên lớp hay số sản phẩm kỳ vọng. Số kỳ vọng chỉ để
người xem đối chiếu. Model TensorRT thường trú phân loại từng crop. Nguồn video
và RTSP tiếp tục dùng giải mã NVDEC của DeepStream. Đây là video tổng hợp từ ảnh.

## Ảnh trung gian thật

1. Frame màu được phân tích ở 320×180.
2. Chuyển ảnh xám.
3. Hiệu màu RGB với nền, bù thay đổi đồng đều bằng median từng kênh.
4. Nhị phân hóa theo ngưỡng gate (mặc định 30).
5. Opening/closing 3×3 và lấp vùng nền kín.
6. Thành phần liên thông, ghép Hungarian có dự báo theo PTS, ROI và track ID.

Mask được lấy từ chính ma trận dùng để quyết định, cập nhật tối đa 5 lần/s;
ghi frame nguồn, PTS và lượt CPU để đối chiếu. Preview vẫn chạy độc lập tối đa
30 FPS. Bật/tắt **Hiện mask**, hoặc bấm **Tải khung xử lý** để lấy bộ sáu ảnh
cùng frame. Mask có bộ nhớ đệm giới hạn, không tự ghi toàn bộ frame ra đĩa.
Overlay live ước lượng vị trí giữa các lượt tracking; mask là frame đã ghi PTS.

## Ý nghĩa bộ đếm

| Chỉ số | Phạm vi |
|---|---|
| CPU · lượt phân tích phiên | Frame thực sự chạy qua gate, tính cả hiệu chuẩn, tối đa 10 lượt/s |
| Chip ảnh xám/trừ nền/ngưỡng/opening/thành phần/chất lượng | Số lần thực thi từng bước, không cộng để gọi là số frame |
| ms/lượt | Host giải mã JPEG, gate, debug, lưu/gửi sự kiện ở lượt gần nhất |
| Frame dựng từ ảnh | Frame do bộ dựng ảnh CPU xuất, riêng nguồn ảnh |
| GPU · yêu cầu trong phiên | Số request_id đã được tiếp nhận từ sản phẩm, không tính warmup |
| GPU hoàn tất / đang chờ / lỗi | Theo nhật ký sản phẩm; ảnh cần xem lại vì chất lượng không tăng GPU |
| Worker hiện tại | Suy luận sản phẩm/warmup toàn worker từ lúc khởi động, khác bộ đếm phiên |
| RAM ứng dụng | Working Set/RSS của Python host và đỉnh vòng đời tiến trình, không gồm browser/WSL/worker GPU |
| RAM máy | Tổng RAM vật lý trừ phần còn khả dụng, báo riêng RSS |
| CPU tiến trình | CPU time / wall time, 100% = một lõi logic, có thể vượt 100% |
| VRAM / tải GPU | nvidia-smi toàn thiết bị, có thể gồm ứng dụng khác |
| Track xác nhận / cao nhất | Track có ít nhất 4/5 lượt quan sát và đỉnh trong phiên |

## Nhiều vật thể và giới hạn

Vật thể tách nhau trong cùng frame có track và sự kiện riêng, kể cả cùng cắt
vạch trong một lượt CPU. Mỗi track cần được xác nhận ít nhất 4/5 lượt, ít nhất
80% vùng sản phẩm trong ROI và cắt vạch đúng chiều. Gate chọn crop và kiểm tra
chất lượng riêng cho từng track rồi gửi từng ảnh đến engine thường trú.

Nhiều sản phẩm trong ROI được xử lý bình thường theo các ID riêng. Bản v3.2
đã bỏ nhánh nhập nhóm/cách ly, chế độ mô phỏng và các cảnh báo tương ứng.
Kiểm tra ảnh mờ/tối và trường hợp chưa xác nhận qua ROI vẫn được ghi nhận.
Tracker dùng các thành phần tiền cảnh từ camera cố định. Cần nghiệm thu với
camera, độ sáng và mật độ dây chuyền thật. Tốc độ mô phỏng là pixel/giây,
chưa phải m/s đã hiệu chuẩn.

## Xác minh lại

```powershell
.\run_demo.ps1
.\.venv-demo\Scripts\python.exe scripts/deployment/verify_conveyor_diagnostics.py
```

Bằng chứng mới ở `artifacts/verification/upgrade_v3_2/`: ba vật thể cùng trong
ROI và cùng cắt vạch → ba lượt GPU; hai vật thể đồng thời, chiều ngược → ba ảnh
qua hai nhóm, ba GPU. Preview HTTP được đo trong cùng lượt thử. Các hồ sơ
kiểm chứng phiên bản cũ được giữ để truy nguyên theo phiên bản.
Đây là kiểm thử chức năng từ ảnh validation, không phải benchmark chính xác
đa vật thể/camera. Model và kết quả nghiên cứu đóng băng không được huấn luyện lại.

Ảnh người dùng tải lên, SQLite và khung vận hành trong `artifacts/inspection`
chỉ nằm ở máy này, không được đưa vào ZIP bàn giao.
