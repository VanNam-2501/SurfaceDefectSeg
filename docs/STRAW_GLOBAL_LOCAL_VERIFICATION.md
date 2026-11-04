# Hồ sơ kiểm chứng nâng cấp CNN toàn cục–cục bộ

Ngày 18-10-2026. Không chưng cất. Mục tiêu của đợt này là biến đề cương thành
pipeline chạy được, kiểm tra learning signal và tạo điều kiện kiểm chứng kết
luận; không tự tuyên bố đạt yêu cầu của trường hoặc phương pháp đã cải thiện.

## Đã triển khai

- src/straw_global_local/data.py: protocol ba tập, kiểm tra hash/nhãn/nhóm,
  view cache lossless từ ảnh gốc, augmentation nhất quán giữa các view.
- model.py: B0–B6 với shared ResNet18, mean/attention/global-context/position.
- runner.py: train/resume, Val-only selection, freeze trước Test, test cache
  kiểm tra provenance, benchmark đầu-cuối và báo cáo readiness.
- Config confirmatory 21 run; pilot riêng không được truy cập Test.
- Notebook Kaggle và source-only bundle độc lập với pipeline hai tầng cũ.

## Đã kiểm chứng thực tế

1. **22 kiểm thử đạt** trong môi trường local. Bao gồm gradient cả encoder
   và thành phần vị trí, không cần nhãn/thương hiệu khi forward, xử lý crop/
   flip, tách nhóm, chống sửa artifact, tích lũy gradient window ngắn, resume
   đúng trạng thái và chặn Test khi chưa đủ điều kiện.
   Log: artifacts/verification/global_local_tests.log.
2. **Smoke đủ bảy cấu hình** trên ảnh thật subset 10 Train / 11 Val / 10
   Test, random init, một epoch, một seed. Train, freeze, test, benchmark và
   report đều hoàn tất. Đây là kiểm chứng kỹ thuật; Test subset smoke đã được
   dùng, không dùng điểm đó như kết quả nghiên cứu. Mười ảnh này được lấy từ
   split Test nguồn; không được mô tả toàn bộ 299 ảnh như holdout chưa từng
   được tiếp xúc. Dataset vốn đã có lịch sử phát triển từ trước.
   Artifact: artifacts/straw_global_local/smoke_v1.
3. **Protocol chính mới**: 1.401 Train / 300 Val / 299 Test, kế thừa nhóm ảnh
   và giữ Val/Test cũ. Không sửa manifest hoặc checkpoint cũ. Cache xác minh
   SHA và kích thước của đủ 2.000 ảnh gốc.
4. **Pilot pretrained thật**: B0 và B5, ba epoch, seed42, cùng protocol mới,
   effective batch32, cùng LR/optimizer/augmentation và BN policy.
   - B0 global224: best Val macro-F1 0,9671695.
   - B5 global-local/context: best Val macro-F1 0,9571265 (xem JSON gốc để
     lấy đầy đủ độ chính xác số).
   - Các mô hình pilot chưa được đánh giá trên tập Test chính. Một seed/ba
     epoch không đủ kết luận ưu thế;
     hiện không có bằng chứng pilot cho rằng thêm local/context giúp hơn B0.
   Artifact: artifacts/straw_global_local/pilot_v1.
5. **B1 global512 đã hoàn tất pilot riêng**: ba epoch/seed42, best Val
   macro-F1 xấp xỉ 0,9808, cao hơn B0/B5 trong lần chạy ngắn này. Không dùng
   kết quả này để tuyên bố ưu thế cuối cùng; Val đã phục vụ phát triển.
   Artifact: artifacts/straw_global_local/pilot_highres_v1.
6. Benchmark cả B0/B1/B5 trên checkpoint pilot: batch1, warmup5/repeats30,
   cùng GPU, có end-to-end và forward riêng. Report development vẫn NOT_READY.
   Bảng số và hình learning curves đã sinh bằng script chỉ đọc Validation:
   artifacts/verification/global_local_pilot/validation_pilot_summary.csv,
   validation_error_cases.csv, pilot_summary.json, validation_learning_curves.png.
   Hình đã được xem kiểm tra nhãn/trục/layout; không dùng hình như kết quả Test.

Pilot hiện ủng hộ việc giữ baseline độ phân giải cao làm đối chứng bắt buộc.
Chưa có tín hiệu để chốt global-local là phương pháp tốt hơn; cần pilot đủ
ngân sách và nhiều seed, hoặc báo kết quả âm nếu không có cải thiện. Không
chạy Test chính để chọn cách sửa kiến trúc.

## Bổ sung gói code Kaggle có resume (18/10/2026)

- `tests/test_global_local.py`: 22 kiểm thử PASS; mã mô hình/train không đổi.
- `tests/test_kaggle_session.py`: 15 kiểm thử PASS, gồm chạy đủ lịch 21 lượt
  bằng mô hình nhỏ, không train lại lượt đã xong, dừng theo giờ, đóng và khôi
  phục ZIP, từ chối checkpoint bị sửa/bundle sai, dò dataset nhiều cấu trúc
  và kiểm tra lại ảnh gốc dù đã có cache.
- Thí nghiệm ngắt sau epoch 1 rồi khôi phục bằng archive cho trọng số,
  optimizer và scheduler khớp chạy liên tục trong cùng môi trường CPU,
  có dropout và augmentation. Không suy rộng thành bit-for-bit trên GPU khác.
- Luồng materialize → prepare → cache → dừng → ZIP → khôi phục → train/report
  được kiểm tra với dữ liệu giả và mô hình nhỏ; mount dataset được đổi tên
  giữa phiên mà fingerprint protocol vẫn giữ nguyên. Trên Windows không có
  quyền tạo symlink, kiểm thử tích hợp dùng alias bằng bản sao ảnh giả nhỏ;
  lời gọi symlink Linux được kiểm tra riêng bằng mock. Chưa chạy trên Kaggle.
- Gói `straw_kaggle_code.zip` chỉ có code/config/CSV split/notebook/docs/tests,
  không có dataset, ảnh gốc, trọng số pretrained hoặc checkpoint huấn luyện.
- Hướng dẫn: [KAGGLE_ALL_IN_ONE.md](KAGGLE_ALL_IN_ONE.md). File resume chỉ được
  giữ qua phiên mới khi Output đã được Kaggle lưu; không tự đồng bộ cloud.

## Nghiên cứu còn thiếu

- Rà định nghĩa nhãn, nhóm ảnh và hồ sơ nguồn/quyền sử dụng đầy đủ; các cờ
  review.json vẫn false. Không bịa reviewer hoặc xác nhận tự động.
- Đối chiếu toàn văn các công trình sát và tính mới cụ thể.
- Thực nghiệm xác nhận B0–B6/ba seed; chưa có main freeze/Test/report cuối.
- Phân tích khoa học đầy đủ các cơ chế/thất bại và luận văn hoàn chỉnh.
- Notebook đã kiểm tra cú pháp; chưa thực thi trên Kaggle hoặc clean install.

## Cách diễn giải readiness

Smoke/pilot report hiện ghi NOT_READY_FOR_THESIS_CONCLUSIONS là đúng.
READY_FOR_INTERNAL_THESIS_ANALYSIS chỉ xác nhận đủ artifact để phân tích;
không thay thế đánh giá khoa học, định nghĩa nhãn, novelty hoặc duyệt của trường.
Điều kiện chi tiết nằm trong STRAW_COMPLETION_REQUIREMENTS.md.
