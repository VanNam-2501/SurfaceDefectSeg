# Báo cáo kiểm thử và xác minh

Ngày chạy: 2026-08-19. Máy local Windows; các phép kiểm tra CUDA VMamba/Kaggle
không chạy lại trong phiên cleanup này.

## Dataset protocol preflight

Lệnh:

```powershell
E:\Project\TTTN\.venv\Scripts\python.exe .\scripts\verification\check_protocol.py `
  --dataset-root E:\Project\TTTN\data\3cad_ani
```

Kết quả:

- Train: 5.733 ảnh.
- Validation: 718 ảnh.
- Test: 717 ảnh.
- Tổng: 7.168 ảnh.
- Split overlap, content overlap và file checks: PASS.
- SHA256 CSV:
  - Train: `9ba4ba9cf7d6e02b189fe58d7bed8111042a72b34773ea37179c0d03474f58e5`
  - Validation: `b33791cc0789603625b5e010bd83c68a92fe0bcf49e65c5548ed90fd7f8c85fd`
  - Test: `cf0372bd3dc6b9c5c584895bc5f77b47b59749cb60b7dc4951dea1de6dbf8604`

## CPU unit tests

| Test | Số test | Kết quả |
|---|---:|---|
| `test_decision_policy.py` | 3 | PASS |
| `test_training_state.py` | 4 | PASS |
| `test_learned_decision_verifier.py` | 3 | PASS |
| Dataset Review Studio | 5 | PASS |
| Tổng | 15 | PASS |

Các test kiểm tra ROI biên tối, PASS/REVIEW/DEFECT của spatial policy, đồng
thuận theo vị trí, phục hồi best epoch, ràng buộc FNR, threshold triage và việc
hybrid không loại bỏ consensus đáng tin.

## Web demo

Lệnh:

```powershell
cd E:\Project\TTTN\apps\web_demo
npm test
```

Kết quả:

- Production build: PASS.
- Server-rendered product test: PASS.
- Health/model availability client wiring test: PASS.
- Tổng: 2 test PASS.
- `apps/web_demo/build/sites-vite-plugin.ts` được xác nhận là source bắt buộc và đã
  được phục hồi từ cleanup quarantine.
- `verify.ps1` đã được sửa để mọi exit code khác 0 từ Python/npm
  làm verification thất bại; không còn trường hợp npm lỗi nhưng cuối cùng vẫn
  in PASS.

Health check với các đường dẫn mặc định mới:

- U-Net checkpoint: available, policy compatible.
- VMamba checkpoint: available, policy compatible.
- SegFormer checkpoint: available, không thuộc final U-Net+VMamba policy.
- Spatial policy: ready.
- Learned hybrid verifier: ready.
- Chế độ: `hybrid_fully_automatic`.

## Visualizations

- 20/20 PNG mở và `Pillow.verify()` PASS.
- 19 SVG được tạo.
- `chart_manifest.csv` có 20 dòng.
- `index.html` tồn tại và liên kết tới hình/bảng tương đối.

## Chưa được xác minh trong phiên này

1. VMamba CUDA selective-scan trên Kaggle GPU.
2. Full end-to-end inference của API với một ảnh sau khi đổi launcher sang
   final U-Net+VMamba policy. Health/config đã PASS nhưng chưa chạy forward ảnh
   trong phiên cleanup.
3. Cài đặt source release trên máy/thư mục sạch.
4. Tái tạo toàn bộ train từ đầu.
5. Dataset review export round-trip sau lần cleanup hiện tại.

Chạy `verify.ps1 -IncludeWeb` trước mỗi lần đóng gói. Lưu terminal
output và commit hash của lần chạy cuối cùng.
