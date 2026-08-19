# Checklist tuân thủ và sản phẩm phải nộp

Ngày audit: **2026-08-19**. Trạng thái phản ánh file thực tế trong
`E:\Project\TTTN`, không dựa trên lời kể.

## Sản phẩm phải nộp

| Sản phẩm | Trạng thái | Bằng chứng hiện có | Việc còn làm |
|---|---|---|---|
| Báo cáo thực tập tốt nghiệp đúng biểu mẫu | **THIẾU** | Chưa có DOCX/PDF báo cáo | Xin đúng template của khoa, viết và xuất PDF |
| Slide bảo vệ | **THIẾU** | Chưa có PPTX/PDF slide | Tạo slide vấn đề → giải pháp → sản phẩm → kết quả → đóng góp |
| Toàn bộ mã nguồn | **CÓ NHƯNG CHƯA ĐÓNG GÓI** | Pipeline ML, demo web, review tool đều có | Tạo source release không chứa dataset/cache/secret |
| Lịch sử quản lý phiên bản | **MỘT PHẦN** | Repo gốc nhánh `main` được chuẩn hóa từ 2026-08-19 | Duy trì commit thật từ baseline; lịch sử trước baseline không được giả mạo |
| Video demo dự phòng | **THIẾU** | Chưa có MP4/MOV/MKV | Quay video theo kịch bản trong `04_demo_video/README.md` |
| Nhật ký thực tập | **THIẾU** | Chưa có hồ sơ | Điền mẫu `05_logs/INTERNSHIP_LOG.md` và ký/xác nhận nếu biểu mẫu yêu cầu |
| Nhật ký phát triển | **MỘT PHẦN** | Có log chạy rời rạc nhưng chưa thành nhật ký | Điền `05_logs/DEVELOPMENT_LOG.md`, dẫn commit và kết quả test |
| AI Development Log | **MỘT PHẦN** | Đã tạo bản hồi cứu các phiên chính | Bổ sung exact model/version, raw thread export và commit cho từng mục |
| Xác nhận đơn vị thực tập | **THIẾU** | Chưa có bản scan | Xin chữ ký/con dấu và đặt vào `06_company_confirmation/` |

## Hồ sơ sử dụng AI bắt buộc

| Trường bắt buộc | Trạng thái | Ghi chú |
|---|---|---|
| 1. Công cụ và phiên bản/tên model | **MỘT PHẦN** | Biết Codex/GPT-5 family; phải chép exact model ID/build từ UI |
| 2. Ngày, mục tiêu, ngữ cảnh | **CÓ BẢN HỒI CỨU** | Đã ghi các phiên chính 2026-08-15 đến 2026-08-19 |
| 3. Prompt gốc và prompt hiệu chỉnh | **MỘT PHẦN** | Có prompt tiêu biểu; vẫn cần export toàn bộ task/chat |
| 4. File/thành phần mã liên quan | **CÓ** | Đã map theo từng phiên chính |
| 5. Kết quả AI trả về | **CÓ BẢN TÓM TẮT** | Nên đính kèm raw response quan trọng |
| 6. Phần chấp nhận/chỉnh sửa/loại bỏ | **CÓ BẢN TÓM TẮT** | Tiếp tục cập nhật ngay sau mỗi phiên |
| 7. Lý do chỉnh sửa | **CÓ BẢN TÓM TẮT** | Gắn với yêu cầu VRAM, FPR/FNR và UX |
| 8. Cách kiểm thử/xác minh | **CÓ** | Xem `07_evidence/VERIFICATION_REPORT.md` |
| 9. Commit tương ứng | **MỘT PHẦN** | Phiên chuẩn hóa có commit thật; các phiên hồi cứu trước baseline không có commit tương ứng |

## Khả năng giải trình

- **Chưa thể đánh dấu hoàn tất bằng file.** Sinh viên phải tự giải thích được
  dataset split, loss, sliding-window inference, threshold Validation, các metric,
  adaptive component rule, learned verifier và hybrid U-Net + VMamba.
- Dùng `03_source/README.md` làm bản đồ học mã.
- Thực hành ít nhất một thay đổi trực tiếp trước buổi bảo vệ: đổi ràng buộc FNR,
  chạy lại Validation selection, chạy test và giải thích ảnh hưởng lên FPR.

## Sai lệch kỹ thuật phải sửa trong báo cáo

1. Protocol cũ ghi effective batch = 4, nhưng checkpoint thực tế là U-Net 38,
   SegFormer 32 và VMamba 16. Phải báo cáo số thực tế.
2. E6 chưa đủ bằng chứng training-time đồng nhất; không đưa vào bảng so sánh
   chính nếu chưa bổ sung đúng điều kiện.
3. Đã xác định dataset là **3CAD**, nhóm **Aluminum New Ipad (ANI)**, và đã
   bổ sung nguồn bài báo/repository trong `07_evidence/DATASET_PROVENANCE.md`.
   Tuy nhiên, repository chính thức chưa nêu license dữ liệu rõ ràng; phải lưu
   thêm điều khoản tải dữ liệu hoặc xác nhận quyền dùng/công bố trước khi phát
   hành dataset ra ngoài.
4. Tên “Learned SegFormer/VMamba” dễ gây hiểu nhầm: score cuối được selector
   chọn là `top_4_mean`, không phải HGB. Chỉ U-Net và fusion ba model dùng HGB.

## Điều kiện “sẵn sàng nộp”

Chỉ đánh dấu sẵn sàng khi không còn mục **THIẾU**, mọi phiên mới trong AI log có commit thật,
`verify.ps1 -IncludeWeb` PASS và source release được thử trên một
thư mục/máy sạch.
