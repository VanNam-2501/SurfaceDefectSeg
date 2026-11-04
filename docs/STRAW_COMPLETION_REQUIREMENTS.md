# Điều kiện đủ bằng chứng để viết kết luận khóa luận

Phạm vi hiện hành: CNN phân loại 5 trạng thái trên dataset ống hút hiện có;
không chưng cất, không lấy thương hiệu hoặc REVIEW làm trọng tâm. Tiêu chí
dưới đây là yêu cầu nghiên cứu nội bộ; không thay thế rubric/duyệt của trường.

## 1. Đề cương được khóa

- Tên: Nghiên cứu kết hợp đặc trưng toàn cục và cục bộ trong CNN cho phân loại
  khuyết tật ống hút.
- RQ1: Độ phân giải toàn ảnh ảnh hưởng thế nào đến từng loại lỗi?
- RQ2: Local và global có bổ sung nhau không?
- RQ3: Attention, ngữ cảnh và vị trí có đóng góp riêng gì so với tổng hợp đơn giản?
- Đánh giá thực dụng: mức cải thiện có đáng chi phí xử lý nhiều view không?
- Đầu ra: Good/Bent/Loose/Missing/Pierced. Quyết định tốt/lỗi suy từ cùng
  argmax 5 lớp; sai loại lỗi không đồng nhất bỏ sót sản phẩm lỗi.

Đóng góp dự kiến là thiết kế và phân tích thực nghiệm. Không gọi việc dùng
CNN/attention/crop là thuật toán hoàn toàn mới trước khi đối chiếu related work.

## 2. Bằng chứng bắt buộc

| Mã | Điều kiện | Artifact / bằng chứng |
|---|---|---|
| D1 | Định nghĩa nhãn rõ, xác minh trường hợp khó/đa lỗi | Bảng quy tắc nhãn, quyết định rà và người rà |
| D2 | Rà nhóm ảnh liên quan, ghi giới hạn metadata | CSV quyết định nhóm/cặp, split và hashes |
| D3 | Nguồn/điều kiện sử dụng được ghi nhận | Trích nguồn và hồ sơ quyền sử dụng |
| L1 | Related work phân biệt kế thừa/khác biệt/claim | Bảng công trình, baseline và giới hạn novelty |
| M1 | Baseline toàn ảnh đủ mạnh được train đầy đủ | B0/B1, lịch train, checkpoint, learning curves |
| M2 | Ablation tách từng thành phần | B2/B3/B4/B5/B6 theo cùng protocol |
| E1 | Tối thiểu 3 seed cho mỗi cấu hình lõi | Config và completion markers của 21 run dự kiến |
| E2 | Chọn checkpoint/family chỉ bằng Val, khóa trước Test | freeze.json và checkpoint hashes |
| E3 | Test đầy đủ, sample ID/nhãn/logits khớp | test_predictions.npz, test.json, bảng từng run |
| E4 | Độ bất định và số lỗi thực tế | Mean/SD seed, paired group bootstrap, confusion/per-class |
| C1 | Chi phí đầy đủ cùng môi trường | Decode/view/transfer/forward p50/p95, VRAM, params |
| A1 | Phân tích cơ chế và thất bại | Ảnh cải thiện/suy giảm, đối chiếu từng RQ, giới hạn |

review.json lưu xác nhận D1/D2/D3/L1 có người và ghi chú dẫn bằng chứng.
Phần mềm không thể tự chứng minh nhãn đúng, nhóm độc lập hoặc novelty. Không
đánh dấu true khi chưa làm; ghi chú phải chỉ tới bằng chứng thực tế.

## 3. Ba loại kết luận hợp lệ

1. Có bằng chứng cải thiện: chênh lệch với baseline mạnh nhất đã chọn bằng
   Val, CI/seed hỗ trợ, và mức lợi ích/chi phí được phân tích.
2. Chưa có bằng chứng rõ: CI chứa 0 hoặc hiệu ứng thiếu ổn định; không đổi
   cách diễn đạt thành "tương đương" nếu chưa có thiết kế kiểm định tương đương.
3. Phương pháp kém hơn: báo baseline thắng, phân tích vì sao và chọn mô hình
   ứng dụng phù hợp. Kết quả âm không được loại khỏi báo cáo.

Sự đầy đủ bằng chứng không yêu cầu B6 phải thắng. Hiệu năng tốt không thay
thế rà dữ liệu; nhiều test phần mềm đạt không thay thế thực nghiệm mô hình.

## 4. Báo cáo tự động và phần phải viết bằng phân tích

Lệnh report tạo report/readiness.json và report/report.md. Chỉ khi purpose
là confirmatory, data không là smoke, đủ B0–B6/3 seed, có review, train đủ,
selection khóa, test khớp và benchmark đầy đủ thì trạng thái tự động mới là
READY_FOR_INTERNAL_THESIS_ANALYSIS. Đây là mức đủ artifact để phân tích, chưa
là phê duyệt khóa luận, chứng nhận novelty hay bảo đảm cải thiện.

Lưu ý: A1 là phân tích khoa học, cần đọc kết quả và viết luận văn. Trạng thái
tự động chưa thay thế A1. Báo cáo thí nghiệm vẫn ghi giới hạn này.

Pilot purpose=development chỉ dùng Train/Val; không được freeze/test. Smoke
kiểm tra luồng code, có thể dùng test nhỏ kỹ thuật nhưng luôn NOT_READY.

## 5. Việc ưu tiên tiếp theo

1. Hoàn tất D1/D2/D3/L1 bằng dữ liệu thật, đồng thời chạy pilot Train/Val.
2. Dùng pilot để khóa recipe và kiểm tra hiệu quả global-local; nếu cần sửa
   code/config thì tạo output/cache version mới, không dùng Test để quyết định.
3. Chạy confirmatory B0–B6, ba seed, benchmark, freeze, test và report.
4. Viết kết luận dựa trên bằng chứng, đối chiếu cả trường hợp không cải thiện.
