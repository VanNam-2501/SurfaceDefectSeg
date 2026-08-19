# Mã nguồn và lịch sử phiên bản

## Trạng thái hiện tại

- Thư mục gốc `E:\Project\TTTN` là Git repository duy nhất, nhánh `main`; lịch
  sử bắt đầu trung thực từ baseline `d91f6fd` ngày 2026-08-19.
- Metadata Git lồng của web/VMamba đã được loại khỏi cây hoạt động để tránh
  repository con ngoài ý muốn.
- `src/threecad_segmentation/third_party/VMamba` được vendor từ upstream commit
  `2ed52ead062a51a64521ed3871d52914bf532876`, có sửa local 15 dòng trong
  `vmamba.py` để import được khi không có Triton.

Lịch sử trước ngày chuẩn hóa không thể tái tạo thành commit thật. Các phiên cũ
được ghi hồi cứu trong AI log; mọi thay đổi từ baseline phải có commit thật.

## Cách hoàn thiện đúng

1. Dùng repository gốc duy nhất, không khởi tạo `.git` trong app/vendor.
2. Không commit `.venv`, dataset, checkpoint, ZIP Kaggle, `node_modules`, cache
   hoặc gallery lớn; `.gitignore` gốc đã được bổ sung.
3. Giữ VMamba dưới dạng vendor dependency có commit nguồn, LICENSE và patch.
4. Commit các mốc tiếp theo bằng nội dung thật, ví dụ:
   - `docs: add compliance checklist and AI development log`
   - `test: add dependency-free verifier unit tests`
   - `fix: make demo use local final checkpoints and U-Net+VMamba policy`
   - `docs: add report and slide artifacts`
5. Điền hash commit vừa tạo vào AI log tương ứng.

Không tạo commit giả có ngày quá khứ để làm như đã có lịch sử. Trong báo cáo
nên nói rõ lịch sử được chuẩn hóa từ thời điểm nào.

## Bản đồ mã cần giải thích được

| Phần | File chính |
|---|---|
| Dataset và patch sampling | `src/threecad_segmentation/training_data.py`, `ani_dataset.py`, `patch_sampler.py` |
| Loss | `src/threecad_segmentation/losses.py` |
| Training | `src/threecad_segmentation/train_common.py`, `scripts/training/train_on_kaggle.py` |
| U-Net | `src/threecad_segmentation/unet_r18.py` |
| SegFormer | `src/threecad_segmentation/segformer_b0.py` |
| VMamba | `src/threecad_segmentation/vmamba_t.py`, `third_party/VMamba/` |
| Full-resolution evaluation | `src/threecad_segmentation/fullres_eval.py`, `scripts/evaluation/evaluate_model.py` |
| Validation threshold | `src/threecad_segmentation/fullres_eval.py` và kết quả E7 |
| Adaptive rule | `src/threecad_segmentation/adaptive_component_policy.py` |
| Learned verifier | `src/threecad_segmentation/learned_decision_verifier.py` |
| Chạy toàn bộ decision experiments | `scripts/experiments/run_all_decision_experiments.py` |
| Audit label/data | `scripts/data/audit_full_dataset_labels.py` |
| Demo API | `apps/web_demo/backend/app.py` |
| Demo UI | `apps/web_demo/app/page.tsx` |
| Review tool | `apps/dataset_review/` |
