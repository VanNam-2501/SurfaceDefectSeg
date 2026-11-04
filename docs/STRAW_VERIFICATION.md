# Trạng thái bàn giao pipeline ống hút

Ngày xác minh: **2026-10-18**. Đây là kiểm chứng mã và smoke test, không phải
kết quả thực nghiệm khóa luận đầy đủ.

## Đã thực hiện

- Audit 2.000 ảnh thuộc 8 thương hiệu, 5 lớp; 0 ảnh hỏng/không nhận diện lớp,
  0 bản trùng nội dung RGB hoàn toàn; 76 cặp ứng viên dHash<=2.
- Gom ứng viên theo thành phần liên thông: 1.943 nhóm; giữ nhóm trong cùng tập.
- Protocol nội bộ: Train 1.201 / Validation 300 / Calibration 200 / Test 299.
- Tạo và kiểm tra manifest cho đủ 8 fold giữ riêng từng thương hiệu.
- Kiểm tra bộ lập lịch: core 27 run, extended 48 run bao gồm core, LOBO 24 run.
- `verify_straw.ps1`: **21 test protocol/huấn luyện/thống kê + 7 test decision
  cũ đều PASS**. Có kiểm thử gián đoạn giữa epoch rồi tiếp tục, so sánh chính
  xác trọng số/scheduler/best epoch với chạy liên tục trên mô hình CPU nhỏ.
- Smoke GPU thật trên RTX 3050 Laptop 4 GB: 3 run ResNet18, 1 epoch,
  random initialization, Train 20 / Validation 10 / Calibration 10 / Test 10.
- Smoke đã hoàn thành train → calibrate → freeze → test → benchmark → report.
  Test gồm clean và noise_2, 7 phương án quyết định; không dùng để kết luận
  model hoặc policy nào tốt hơn.
- Chạy lại train/test: nhận diện và bỏ qua các bước đã hoàn thành, kiểm tra hash.
- Notebook: các cell Python parse thành công; gói ZIP source được tạo riêng,
  không kèm ảnh gốc/checkpoint/cache.
- Đã mở kiểm tra biểu đồ so sánh do smoke sinh ra.

## Bằng chứng local

- Audit: `artifacts/straw_thesis/data_v1/audit.json` và
  `artifacts/straw_thesis/data_v2/audit.json`.
- Protocol chính: `artifacts/straw_thesis/data_v2/iid/protocol.json`.
- Fold thương hiệu: `artifacts/straw_thesis/lobo_data_v1/`.
- Test log: `artifacts/verification/straw_tests_latest.log`.
- Smoke: `artifacts/straw_thesis/smoke_v2/report/index.html`;
  `report/status.json` ghi **SMOKE ONLY — NOT THESIS RESULTS**.
- Plan chính/mở rộng: `artifacts/straw_thesis/core_v1/plan.json` và
  `artifacts/straw_thesis/extended_v1/plan.json`.
- Gói tải lên Kaggle: `artifacts/packages/straw_thesis_source.zip`.

## Chưa hoàn thành

- Chưa chạy 27/48 run chính và 24 run LOBO trên Kaggle.
- Chưa kiểm chứng notebook trong phiên Kaggle thực tế; tài khoản, GPU và dữ
  liệu gắn vào notebook cần được thiết lập trên Kaggle.
- Chưa có kết quả accuracy/FNR/REVIEW mới đủ điều kiện viết kết luận khóa luận.
- Chưa rà soát trực quan đầy đủ 76 cặp ảnh ứng viên hoặc xác định phiên chụp.
- Chưa có tập ảnh ngoài độc lập, chưa xác nhận khả năng vận hành trên nhà máy.
- Chưa tích hợp policy/checkpoint nghiên cứu mới vào demo cũ; không thay demo
  bằng checkpoint smoke.

Dataset hiện tại đã được dùng trong phát triển trước đây. Chia lại dữ liệu
không khôi phục tính độc lập của một tập kiểm chứng chưa từng được quan sát.
Hướng dẫn phương pháp, lệnh chạy và lộ trình nằm tại `docs/STRAW_THESIS.md`.
