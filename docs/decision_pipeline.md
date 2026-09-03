# Decision pipeline local: giảm báo động giả

Pipeline này tách rõ hai bước:

1. Model tạo probability map.
2. Decision policy biến probability thành `PASS / REVIEW / DEFECT`.

Policy chỉ được hiệu chỉnh trên **Validation**, lưu thành
`artifacts/experiments/decision/policy/decision_policy.json`, sau đó giữ nguyên khi chạy
**Test** và web demo.

## Logic đã tích hợp

- Loại vùng gần như đen nối với biên ảnh (padding/gá đen ở ngoài sản phẩm).
- Gom pixel thành connected components.
- Vùng đủ lớn là strong component.
- Vùng nhỏ nhưng xác suất rất cao đi vào `REVIEW`, không bị xóa thẳng.
- Nhiều model phải đồng ý ở gần cùng vị trí mới kết luận `DEFECT`.
- Model bất đồng chuyển sang `REVIEW`.
- Mục tiêu mặc định: alert FNR không quá 2%, auto-Defect recall tối thiểu 80%.
- Test không được dùng để thay đổi policy.

## Chạy với U-Net và SegFormer hiện có

Mở PowerShell tại `E:\Project\TTTN`:

```powershell
.\run_decision_pipeline.ps1 -Action All
```

Lệnh `All` lần lượt:

1. Xuất probability cho Validation và Test. U-Net đã có cache nên các PNG cũ
   được bỏ qua; SegFormer sẽ được chạy một lần.
2. Quét threshold, minimum component area và mức đồng thuận trên Validation.
3. Khóa `decision_policy.json`.
4. Đánh giá policy đã khóa trên Test.

Có thể chạy từng bước để dễ theo dõi:

```powershell
.\run_decision_pipeline.ps1 -Action Export
.\run_decision_pipeline.ps1 -Action Calibrate
.\run_decision_pipeline.ps1 -Action Test
```

Kết quả chính:

```text
artifacts/experiments/decision/
├── policy/
│   ├── decision_policy.json
│   ├── unet_policy_scan.csv
│   ├── segformer_policy_scan.csv
│   └── ensemble_policy_scan.csv
└── test/
    ├── decision_metrics.json
    ├── per_image_decisions.csv
    ├── decision_confusion.csv
    └── defect_group_decisions.csv
```

## Thêm VMamba sau khi Kaggle train xong

Tải toàn bộ run VMamba về local, sau đó chạy lại `All` với checkpoint mới:

```powershell
.\run_decision_pipeline.ps1 -Action All `
  -VmambaCheckpoint "E:\duong_dan_vmamba\checkpoints\best.pt"
```

Probability U-Net/SegFormer đã có sẽ được bỏ qua. Pipeline chỉ xuất thêm
VMamba, rồi hiệu chỉnh lại policy cho đủ ba model và test lại policy mới.

## Chạy web demo

Sau khi đã có `decision_policy.json`:

```powershell
.\start_decision_demo.ps1
```

Khi có VMamba:

```powershell
.\start_decision_demo.ps1 `
  -VmambaCheckpoint "E:\duong_dan_vmamba\checkpoints\best.pt"
```

Web hiển thị quyết định cuối, lý do, số model đồng thuận, mask cuối và kết quả
từng model. Thanh threshold thủ công đã được thay bằng policy đã khóa để tránh
vô tình chỉnh theo một ảnh hoặc theo Test.

## Điều chỉnh mục tiêu vận hành

Ví dụ yêu cầu alert FNR tối đa 1%, Defect recall tối thiểu 85%:

```powershell
.\run_decision_pipeline.ps1 -Action Calibrate `
  -FnrLimit 0.01 `
  -MinDefectRecall 0.85

.\run_decision_pipeline.ps1 -Action Test
```

Nếu constraint không đạt, `selection_status` trong policy sẽ ghi rõ constraint
nào phải được nới hoặc fallback nào đã được dùng; pipeline không âm thầm tuyên
bố rằng mục tiêu đã đạt.
