# Sửa tracking v3.3 — 04/11/2026

## Nguyên nhân đã tái hiện

Trên 39 ảnh đang chọn, nền xám của nhiều ảnh gần màu băng chuyền. Phép trừ
nền xám chỉ giữ một phần hộp/ống hút trong ảnh, làm một ảnh tách thành nhiều
thành phần hoặc quá nhỏ để được giữ lại. Bộ ghép tâm trước đây dùng vận tốc
theo lượt, trong khi khoảng cách PTS giữa các lượt thay đổi khi bỏ khung hình.
Điều kiện kích hoạt cũng chỉ được xét tại đúng một lượt cắt vạch.

## Thay đổi

- Trừ nền màu RGB, bù median theo từng kênh; giữ ảnh xám trong phần minh họa.
- Opening và closing 3×3, sau đó lấp các vùng nền kín bằng thành phần liên thông.
  Nền nối tới biên ảnh được giữ nguyên, nên các sản phẩm tách nhau vẫn tách riêng.
- Ghép một-một bằng Hungarian, kết hợp vị trí dự báo theo PTS, IoU, kích thước
  và khoảng cách theo phương ngang/dọc. Vận tốc tính bằng pixel/giây.
- Giữ ID qua các lượt tạm mất quan sát. Overlay dự báo dùng nét đứt và nhãn
  “dự báo tạm thời”; chỉ quan sát thật mới được chọn ảnh và tạo yêu cầu GPU.
- Ghi nhớ lần cắt vạch tối đa 900 ms để chờ xác nhận 4/5 lượt và ít nhất 80%
  vùng sản phẩm trong ROI. Mỗi track tiếp tục chỉ tạo một bản ghi.

## Đường ảnh chọn trên băng chuyền

Ảnh được đặt trên thẻ có viền màu 15 pixel trong khung 960×540, giúp tách toàn
bộ ảnh khỏi nền băng chuyền. Detector nhận chính các pixel đó. Tọa độ dựng,
ID ảnh và số sản phẩm tham chiếu không được đưa vào gate để kích hoạt AI.
Viền được cắt bỏ bằng quy ước crop của nguồn ảnh (5 pixel ở lưới tracking
320×180), không thêm padding cho nguồn này. CNN và kiểm tra chất lượng nhận
phần ảnh bên trong. Có kiểm thử ảnh phẳng: thẻ vẫn track được nhưng ảnh vẫn bị
đánh dấu mờ/tương phản thấp; viền không làm giả điểm chất lượng.

Khi thẻ mới vào hoặc sắp ra mép hình, chỉ giữ contour ngoài của thẻ để tránh
phần hộp trong ảnh tạo thêm track con. Quy tắc này chỉ áp dụng định dạng thẻ ảnh.

Video/RTSP sử dụng các pixel của nguồn gốc, không có thẻ; giữ padding mặc định.
Các ngưỡng sáng, nét và độ tương phản không bị hạ. Bảng kết quả hiển thị trực
tiếp lý do cần xem lại và “Không gọi GPU”, giúp phân biệt tracking với chất lượng.

## Kiểm chứng

Đo trọn 39 ảnh đã chọn trong replay bằng JPEG thực, một và ba sản phẩm đồng
thời. Dùng lịch dựng chỉ trong bộ đánh giá để đối chiếu crop/PTS sau xử lý;
không truyền lịch này cho tracker. Kết quả:

| Lượt | Sản phẩm ghi nhận | Thiếu / lặp | Crop thiếu | Đổi ID trong ROI |
|---|---:|---:|---:|---:|
| v3.2, một sản phẩm | 23/39 ảnh riêng, 28 bản ghi | 16 thiếu / 5 lặp | 26 | 4 ảnh |
| v3.2, ba sản phẩm | 7/39 | 32 thiếu / 0 lặp | 2 | 0 ảnh quan sát được |
| v3.3, một sản phẩm | 39/39 | 0 / 0 | 0 | 0 |
| v3.3, ba sản phẩm | 39/39 | 0 / 0 | 0 | 0 |

Lượt bổ sung 600 px/s với khoảng cách khung hình không đều cũng đạt 39/39
ở cả hai cấu hình. Đây là so sánh toàn đường demo trước/sau, có thay đổi cách
dựng thẻ ảnh; không phải ablation giữ nguyên từng pixel đầu vào.

Kiểm tra GPU thật ở cổng riêng, 600 px/s: một sản phẩm → 39 lần GPU;
ba sản phẩm → 16 lần GPU và 23 review do độ nét. Cả hai đều ghi nhận đủ 39 ảnh.
IoU nhỏ nhất giữa crop và toàn bộ ảnh tham chiếu lần lượt khoảng 0,987 và 0,969.
Preview HTTP khoảng 30 FPS. Số GPU khác số track khi ảnh không đạt chất lượng.
Các số trên là kiểm tra chức năng của ảnh phát lại, chưa nghiệm thu camera thật
và không dùng làm độ chính xác phân loại. Model/engine giữ nguyên.

Hồ sơ tổng hợp: `artifacts/verification/upgrade_v3_3/`. Ảnh người dùng và chi tiết
riêng ở `artifacts/inspection/tracking_audit/`, được loại khỏi ZIP bàn giao.

```powershell
# Kiểm tra bộ ảnh đang chọn bằng server thử nghiệm đang mở ở cổng 8035
.\.venv-demo\Scripts\python.exe scripts/deployment/verify_tracking_replay.py --url http://127.0.0.1:8035
```
