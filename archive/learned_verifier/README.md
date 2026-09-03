# Learned verifier archive

Thư mục này lưu phần Learned Verifier và Learned Hybrid đã được tách khỏi
pipeline chính vào ngày 2026-08-22.

## Trạng thái

- Chỉ giữ để truy vết và tái kiểm tra thí nghiệm cũ.
- Không được gọi bởi web demo, verification hoặc pipeline thí nghiệm chính.
- Pipeline chính tiếp tục sử dụng ba segmentation model, Adaptive Component
  Policy và spatial rule-based policy.
- Test và artifact cũ không bị xóa; mọi nội dung có thể được khôi phục thủ công.

## Cấu trúc

- src/learned_decision_verifier.py: implementation HistGradientBoosting và
  learned/spatial hybrid.
- tests/test_learned_decision_verifier.py: unit test cũ.
- scripts/run_learned_verifier.ps1: runner cũ.
- artifacts/experiments/learned_verifier/: output thí nghiệm độc lập.
- artifacts/reports/learned_all3/: kết quả learned ba model.
- artifacts/reports/hybrid_pairs/: kết quả hybrid theo từng cặp model.

Các development log và hồ sơ lịch sử trong submission/ vẫn giữ nguyên để phản
ánh đúng quá trình phát triển. Chúng không còn mô tả phạm vi chạy mặc định.
