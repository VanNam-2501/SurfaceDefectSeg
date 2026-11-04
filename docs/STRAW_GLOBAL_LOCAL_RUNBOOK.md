# Chạy nghiên cứu CNN toàn cục–cục bộ

Đây là pipeline mới triển khai B0–B6, không chưng cất. Runner hai tầng và
notebook straw_thesis cũ không chạy mô hình này.

## 1. Các cấu hình

| ID | Mô hình |
|---|---|
| B0 | Global ResNet18 224 |
| B1 | Global ResNet18 512 |
| B2 | Bốn crop 224, mean pooling |
| B3 | Global 224 + bốn crop, mean pooling, ghép đặc trưng |
| B4 | B3 + attention dựa vào local |
| B5 | B4 + ngữ cảnh global trong score |
| B6 | B5 + tọa độ crop |

Mỗi crop chiếm 60% chiều rộng/cao, bốn góc phủ toàn ảnh. Crop được tạo từ
ảnh gốc, không dùng mask hoặc nhãn. Nhãn chỉ giám sát dự đoán tổng hợp.
Không gọi các crop là ROI ống hút đã được định vị. Không gán nhãn ảnh cho
từng crop. Một ảnh với 5 view vẫn là một mẫu dữ liệu.

Cache lưu RGB uint8 lossless của global224/global512/local/coords. Nguồn
được kiểm tra SHA và kích thước gốc. Augmentation dùng cùng brightness cho
các view và whole-image horizontal flip tương ứng bằng flip + đổi thứ tự
crop. Không random crop/erase làm mất sản phẩm. Inference không cần nhãn/thương hiệu.

CNN dùng shared encoder. CE ở cấp ảnh; AdamW, LR encoder/head riêng, cosine,
AMP trên CUDA. BatchNorm mặc định cố định running stats, affine vẫn học,
áp cùng mọi cấu hình. Số bước tích lũy giữ effective batch 32 ảnh, bao gồm
window cuối không đầy. Checkpoint/resume ở ranh giới epoch có RNG/optimizer/
scheduler/scaler. Chạy gián đoạn giữa epoch sẽ chạy lại từ epoch cuối đã lưu.

## 2. Chuẩn bị protocol ba tập

```powershell
.\.venv\Scripts\python.exe -m src.straw_global_local.runner prepare
.\.venv\Scripts\python.exe -m src.straw_global_local.runner cache --cache-workers 4
```

Nguồn mặc định: artifacts/straw_thesis/data_v2/iid. Gộp Train+Calibration
thành 1.401 Train, giữ 300 Val và 299 Test; tạo bản mới, không sửa manifest cũ.
Cache đọc ảnh gốc, không đọc cache384 của pipeline cũ. Nếu chuyển máy/Kaggle,
dùng --image-root trỏ tới thư mục chứa đường dẫn relative_path trong CSV.

Cache có thể tiếp tục sau gián đoạn, xác minh hash từng file đã hoàn thành.
Thay implementation views/protocol cần cache directory mới. Không sửa code
trong khi train rồi dùng lại cùng experiment output.

## 3. Pilot phát triển (không truy cập Test)

```powershell
.\.venv\Scripts\python.exe -m src.straw_global_local.runner train --config configs/straw_global_local_pilot.json --output artifacts/straw_global_local/pilot_v1
.\.venv\Scripts\python.exe -m src.straw_global_local.runner report --config configs/straw_global_local_pilot.json --output artifacts/straw_global_local/pilot_v1
```

Pilot mặc định pretrained B0/B5 một seed, ba epoch. Nó chỉ kiểm tra learning
signal và chi phí ban đầu; không là bảng kết quả cuối hoặc bằng chứng phương
pháp tốt hơn. Đối chứng B1 trên Val có config riêng
configs/straw_global_local_pilot_highres.json và output riêng
artifacts/straw_global_local/pilot_highres_v1; chạy trước khi chốt recipe.
Ghi cả các lần thử, không chỉ cấu hình thuận lợi.

--max-epochs-this-call 1 giới hạn số epoch chạy trong một lần gọi để thử
resume; nó không biến run chưa đủ epoch/patience thành completed.

## 4. Thực nghiệm xác nhận

```powershell
.\.venv\Scripts\python.exe -m src.straw_global_local.runner plan
.\.venv\Scripts\python.exe -m src.straw_global_local.runner train
.\.venv\Scripts\python.exe -m src.straw_global_local.runner benchmark
```

Config chính dự kiến 21 run (7 cấu hình × 3 seed); mặc định 40 epoch trần,
patience10, chọn checkpoint macro-F1 Val rồi CE Val. Có thể chỉ định nhiều
--run-id B0_s42 để chia công việc Kaggle. Mọi run phải dùng cùng code/config/
protocol/cache fingerprints. Thay chúng cần output mới.

Benchmark dùng ảnh Val gốc, batch1, warmup5/repeats30, sync CUDA; báo cả
end-to-end và forward, peak PyTorch allocated (không phải toàn bộ VRAM hệ thống).

Sau khi hoàn tất review.json bằng bằng chứng thực, người rà và ghi chú:

```powershell
.\.venv\Scripts\python.exe -m src.straw_global_local.runner freeze
.\.venv\Scripts\python.exe -m src.straw_global_local.runner test
.\.venv\Scripts\python.exe -m src.straw_global_local.runner report --bootstrap 1000
```

Freeze yêu cầu tất cả run hoàn tất, checkpoint/Val predictions nguyên vẹn,
review được ghi nhận. Family baseline chọn từ B0/B1, family global-local từ
B3–B6 theo macro-F1 Val trung bình seed. Test chỉ mô tả các cấu hình đã khóa,
không chọn mô hình. Test cache bị sửa sẽ bị từ chối, không bị bỏ qua âm thầm.

Đầu ra report: validation_runs.csv, test_runs.csv, test_summary.csv,
benchmarks.csv, readiness.json, report.md. Confusion/per-class nằm trong
runs/*/test.json; logits/labels/sample IDs nằm trong test_predictions.npz.
Chênh lệch chính dùng paired group bootstrap trung bình metric qua seed,
không ensemble logits và không coi seed là ảnh test độc lập.

## 5. Smoke kỹ thuật

```powershell
.\.venv\Scripts\python.exe -m src.straw_global_local.runner prepare --protocol artifacts/straw_global_local/smoke_data_v1 --smoke-per-class 2
.\.venv\Scripts\python.exe -m src.straw_global_local.runner cache --protocol artifacts/straw_global_local/smoke_data_v1 --cache artifacts/straw_global_local/smoke_cache_v1
```

Dùng config straw_global_local_smoke.json, protocol/cache như trên và output
riêng cho train/benchmark/freeze/test/report. Random init, một epoch, một seed.
Smoke luôn NOT_READY_FOR_THESIS_CONCLUSIONS dù code chạy đủ luồng.

## 6. Kaggle và kiểm thử

- Gói code chạy đủ 21 lượt có tự dừng và resume nhiều phiên:
  `artifacts/packages/straw_kaggle_code.zip`; hướng dẫn
  [KAGGLE_ALL_IN_ONE.md](KAGGLE_ALL_IN_ONE.md).
- Notebook mới: `experiments/notebooks/straw_global_local_all_in_one_kaggle.ipynb`.
  Gắn dataset ảnh trực tiếp từ Kaggle; RAW_ROOT tự dò hoặc đặt thủ công.
  Tạo gói bằng `python scripts/research/package_kaggle_all_in_one.py`.
  Resume ZIP chứa cache/checkpoints, không chứa ảnh gốc; phải Add Input Output
  của phiên trước để tiếp tục ở phiên mới. Checkpoint không tự đồng bộ cloud.
- Notebook: experiments/notebooks/straw_global_local_kaggle.ipynb.
- Đóng gói: python scripts/research/package_global_local.py.
- Bundle gồm code/config/docs/test/notebook và CSV protocol nguồn, không có
  ảnh, checkpoint, credentials hoặc review giả.
- Cần attach zip source và dataset ảnh gốc trên Kaggle. Giữ cặp torch/
  torchvision tương thích của Kaggle; pretrained ResNet18 cần Internet hoặc
  weight file chính thức đã cache, không tự fallback random init.
- Kiểm thử: python -m unittest discover -s tests -p test_global_local.py -v.

Các kết luận phải tuân theo STRAW_COMPLETION_REQUIREMENTS.md. Dữ liệu đã được
dùng trong lịch sử phát triển; đây là benchmark nội bộ, không phải xác nhận
trên nhà máy hoặc bộ ảnh độc lập mới.
