# Data Review Studio

Công cụ local-first để rà soát và làm sạch dataset segmentation với số lượng lớn.
Dataset nguồn luôn **chỉ đọc**; mọi quyết định, mask đã sửa và bản export được lưu
riêng trong `apps/dataset_review`.

## Khởi động nhanh

Từ thư mục workspace:

```powershell
.\run_review.ps1
```

Script tự mở `http://127.0.0.1:8765` và tự nạp probability cache cục bộ tại
`artifacts/experiments/decision/predictions` nếu thư mục tồn tại.

Chọn dataset hoặc kết quả model khác:

```powershell
.\run_review.ps1 `
  -DatasetRoot "E:\Project\TTTN\data\3cad_ani" `
  -ResultsRoot "E:\Dowload\results\TTTN\results" `
  -Port 8765
```

Có thể truyền nhiều `ResultsRoot` để đối chiếu UNet, SegFormer và các seed:

```powershell
.\run_review.ps1 -ResultsRoot @(
  "E:\results\unet",
  "E:\results\segformer"
)
```

## Workflow khuyến nghị

1. Lọc `Good score cao`, `False positive`, `Zero overlap` hoặc `Model bất đồng`.
2. Kiểm tra ảnh gốc, mask, overlay và ảnh qualitative của model.

Lưu ý: chọn model chỉ lọc các dòng trong `per_image_metrics.csv`, không tự chạy
checkpoint. Ảnh dự báo chỉ xuất hiện khi lần đánh giá đã lưu PNG tương ứng trong
thư mục `test/qualitative` hoặc `validation/qualitative`. Nếu muốn xem dự báo cho
mọi ảnh, chạy bộ xuất toàn phần dưới đây.

### Xuất dự báo cho toàn bộ dữ liệu

Xuất cả UNet và SegFormer cho `train`, `val`, `test`:

```powershell
cd E:\Project\TTTN
.\apps\dataset_review\export_both_models.ps1
```

Hoặc chỉ chạy một model:

```powershell
.\apps\dataset_review\export_predictions.ps1 -Model unet
.\apps\dataset_review\export_predictions.ps1 -Model segformer
```

Mỗi ảnh được lưu thành probability PNG trong thư mục
`<model-run>/predictions/<split>/probability`. Có thể nhấn `Ctrl+C` để dừng và
chạy lại cùng lệnh; các ảnh đã hoàn tất sẽ được bỏ qua. Sau khi xuất xong, khởi
động lại Data Review Studio. Khung **Dự báo model** cho phép đổi giữa `Overlay`,
`Heatmap` và `Binary`.
3. Chọn một kết luận:
   - **Duyệt — giữ nguyên**: nhãn và mask đúng.
   - **Dấu chấp nhận được**: vẫn là Good, đồng thời có thể đưa vào hard-negative.
   - **Đổi nhãn Good/Defect**: sửa nhãn trong bản export.
   - **Cần sửa mask**: mở mask editor để vẽ/xóa hoặc nạp PNG thay thế.
   - **Chưa chắc**: giữ cho reviewer thứ hai; mặc định không vào bản sạch.
   - **Loại khỏi dataset sạch**: không xuất mẫu này.
4. Gắn issue tags và ghi chú để audit lại được quyết định.
5. Bấm **Xuất dữ liệu sạch** sau khi hoàn tất một batch.

## Phím tắt

- `←` / `→`: ảnh trước / sau.
- `A`: duyệt giữ nguyên.
- `H`: acceptable mark + hard-negative.
- `G` / `D`: đổi thành Good / Defect.
- `M`: mở mask editor.
- `U`: chưa chắc.
- `X`: loại.
- `Ctrl+Enter`: lưu và sang ảnh tiếp.
- Trong mask editor: `B` vẽ, `E` xóa, `Ctrl+Z` hoàn tác.

## Dữ liệu được lưu ở đâu

- `review_state.sqlite3`: trạng thái hiện tại và lịch sử thay đổi.
- `edits/masks/`: mask chỉnh sửa, không ghi đè mask nguồn.
- `exports/<tên_bản_export>/`: manifest/split sạch và audit log.

Mỗi bản export gồm:

- `training_dataset/`: **bộ dữ liệu dùng để train ngay**. Dùng thư mục này làm
  `dataset_root`; các manifest nằm ở
  `training_dataset/dataset_audit/splits/train.csv`, `val.csv`, `test.csv`.
  Mọi `image_path` và `mask_path` trong các CSV này là đường dẫn tương đối
  bên trong `training_dataset`. Ảnh Good có mask PNG đen; ảnh Defect dùng mask
  cuối cùng sau review.  Xem thêm `training_dataset/TRAINING_READY.md`.
- `cleaned_manifest.csv`
- `splits/train.csv`, `val.csv`, `test.csv`
- `corrected_masks/`
- `audit_log.csv`
- `review_events.jsonl`
- `hard_negatives.csv`
- `unresolved.csv`
- `export_summary.json`

Các mẫu `uncertain`, `exclude`, mask defect rỗng hoặc defect mới chưa có mask sẽ
không bị âm thầm đưa vào bản sạch; chúng được ghi vào `unresolved.csv`.

## Chạy kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s .\apps\dataset_review\tests -v
```

## Cài dependency ở môi trường mới

```powershell
python -m pip install -r .\apps\dataset_review\requirements.txt
```
