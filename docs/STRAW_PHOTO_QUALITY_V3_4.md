# Sửa cảnh báo độ nét của băng chuyền ảnh — 04/11/2026

## Nguyên nhân

Ở v3.3, ba sản phẩm được dựng thành các ảnh 128×128 trong khung 960×540.
Kiểm tra chất lượng nhận ảnh crop từ chính khung này rồi resize 256×256 để
đo phương sai Laplacian. Chi tiết đã mất trước phép đo. Cùng 39 ảnh vượt
ngưỡng khi chấm từ ảnh gốc nhưng có 23 ảnh bị chặn độ nét trong replay ba làn.
Việc tracker ghi đủ 39 sản phẩm không kiểm chứng được lỗi ở đường chất lượng.

## Sửa nguồn ảnh

- Dựng nguồn 1920×1080 trực tiếp từ ảnh tải lên; nội dung ảnh ở một/hai/ba làn
  lần lượt 440/380/256 pixel. Không phóng to khung preview cũ để giả độ nét.
- Gate nhận frame RGB nguồn. Crop kiểm tra chất lượng và crop gửi TensorRT
  lấy từ cùng nguồn đó, theo các hộp do detector quan sát từ pixel.
- Preview được thu nhỏ riêng còn 960×540 và nén MJPEG. Preview/thumbnail
  không đi ngược vào đường chấm chất lượng hoặc phân loại.
- Viền thẻ tăng cùng tỷ lệ thành 30 pixel, vẫn bị cắt bỏ trước kiểm tra
  chất lượng và CNN. Lưới tracking vẫn 320×180, inset vẫn 5 pixel.
- Tốc độ nguồn tăng cùng tỷ lệ 2×. Thông số tốc độ người dùng vẫn theo khung
  hiển thị 960 pixel, nên thời gian di chuyển và thời lượng phiên giữ nguyên.
- Ngưỡng nét 18, ngưỡng sáng và tương phản, công thức đo và engine giữ nguyên.
  Đây là ngưỡng kỹ thuật, chưa được hiệu chuẩn thành xác suất chất lượng.

## Truy nguyên và giao diện

Phiên mới lưu `photo-source-v3.4`, kích thước nguồn/preview, kích thước ảnh
sản phẩm và tốc độ nguồn trong cấu hình phiên. Mỗi bản ghi lưu kích thước
nguồn; chất lượng ghi kích thước crop và phương pháp đo. Chi tiết kết quả
hiển thị điểm nét, ngưỡng và kích thước ảnh. Nhãn “Độ nét dưới ngưỡng” mô tả
đúng phép đo; bản ghi bị chặn vẫn hiển thị “Không gọi GPU”.

Lịch sử cũ giữ nguyên để truy nguyên; bản sửa không đổi các bản ghi v3.3
thành kết quả mới. Đọc trường revision/cấu hình phiên khi so sánh.

## Kiểm chứng

Hồ sơ tổng hợp ở `artifacts/verification/upgrade_v3_4/`. Ảnh và chi tiết riêng
của bộ 39 ảnh ở `artifacts/inspection/`, không đưa vào ZIP bàn giao.

Các kiểm thử mới tái hiện được ảnh rõ bị mất chi tiết ở 128 px; xác nhận
ảnh rõ qua ngưỡng tại cả ba chế độ; ảnh Gaussian blur, tối và phẳng vẫn bị
chặn; ba sản phẩm hỗn hợp được track riêng trong cả hai chiều và khi PTS
không đều; kiểm tra phiên thực dùng crop nguồn, preview riêng và GPU chỉ
nhận ảnh đạt chất lượng. Kiểm thử thêm xác nhận thời lượng/tốc độ không đổi.

Kết quả GPU thật, cùng engine `47bcee9a…`, tốc độ 600 px/s theo preview:

| Trường hợp | Bản ghi | Lần GPU hoàn tất | Review chất lượng | Preview HTTP |
|---|---:|---:|---:|---:|
| 39 ảnh, một sản phẩm | 39 | 39 | 0 | 29,92 FPS |
| 39 ảnh, hai sản phẩm | 39 | 39 | 0 | 29,93 FPS |
| 39 ảnh, ba sản phẩm | 39 | 39 | 0 | 29,92 FPS |
| 50 đầu vào, ba sản phẩm, chiều ngược | 50 | 50 | 0 | 29,92 FPS |

Trường hợp 50 dùng bộ 39 ảnh cộng 11 lượt lặp, kiểm tra giới hạn số đầu vào
và nhóm cuối có hai sản phẩm. Các trường hợp đều không thiếu/lặp bản ghi;
IoU crop nhỏ nhất 0,969. Không có yêu cầu GPU lỗi hoặc còn chờ.

Kiểm tra đối chứng bằng ba ảnh (rõ, Gaussian blur, tối) cùng một lượt:
đủ ba bản ghi, chỉ một lần GPU, hai review. Ngưỡng nét vẫn 18 trong tất cả
các lượt. Trọn 41 kiểm thử Python đã qua, cùng kiểm tra JavaScript và gate cũ.

```powershell
# Chạy sau khi đã mở bản kiểm thử ở 8035
.\.venv-demo\Scripts\python.exe scripts/deployment/verify_tracking_replay.py --url http://127.0.0.1:8035 --simultaneous 3 2 1 --speed 600 --expect-reviews 0
.\.venv-demo\Scripts\python.exe tests/test_photo_quality_resolution.py -v
```

Đây là kiểm tra chức năng với ảnh/video phát lại. Không dùng số lượng
ảnh qua ngưỡng hoặc số lần GPU làm độ chính xác phân loại hay nghiệm thu camera.
