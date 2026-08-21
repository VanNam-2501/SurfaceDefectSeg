# Train lại ba model trên Windows

Script `retrain_models.ps1` dùng cùng một protocol cho U-Net/ResNet-18,
SegFormer-B0 và VMamba-T:

1. kiểm tra split và đường dẫn ảnh/mask;
2. train model;
3. chọn probability threshold **chỉ trên Validation**;
4. chạy Test với threshold đã khóa.

Kết quả luôn nằm trong `results_retrained/<model>/<run-name>/`, không ghi đè
lên các kết quả cũ.

## Dataset dùng để train

Mặc định, script dùng:

```text
data/3cad_ani/
```

Sau khi review hoàn tất, cần train bằng dataset đã export từ Data Review Studio,
không phải bằng dataset gốc. Truyền đường dẫn thư mục `training_dataset` của bản
export vào `-DatasetRoot`; thư mục này đã có sẵn ảnh, mask và ba frozen splits.

## Chạy U-Net và SegFormer trên máy hiện tại

Mở PowerShell tại thư mục này:

```powershell
cd E:\Project\TTTN

.\scripts\training\retrain_models.ps1 -Model unet -RunName cleaned_v1_seed42
.\scripts\training\retrain_models.ps1 -Model segformer -RunName cleaned_v1_seed42
```

Với dataset đã review/export:

```powershell
.\scripts\training\retrain_models.ps1 -Model unet `
  -DatasetRoot "E:\Project\TTTN\apps\dataset_review\exports\cleaned_final\training_dataset" `
  -RunName cleaned_v1_seed42
```

Script đặt batch size bằng 1, gradient accumulation bằng 4 và tile batch size
bằng 1. Đây là thiết lập an toàn hơn cho RTX 3050 Laptop 4 GB; effective batch
size vẫn là 4.

## VMamba-T

VMamba đã là model thứ ba trong pipeline và trong web demo. Trước khi train,
script bắt buộc chạy `test_vmamba_runtime.py` để kiểm tra CUDA selective-scan.

Mã backbone tối thiểu nằm tại
`src/threecad_segmentation/third_party/VMamba/vmamba.py`. Wheel đi kèm được đóng
băng cho runtime Kaggle T4 cụ thể và không dùng trực tiếp trên RTX 3050 Windows.
Vì vậy:

- train U-Net và SegFormer trên máy này ngay được;
- train/evaluate VMamba trên Kaggle theo `docs/guides/kaggle_training.md`, hoặc
  tự build `mamba-ssm` tương thích CUDA, PyTorch và GPU Windows của bạn;
- khi VMamba runtime đã pass smoke test, chạy:

```powershell
.\scripts\training\retrain_models.ps1 -Model vmamba -RunName cleaned_v1_seed42
```

Sau khi runtime VMamba hoạt động, có thể chạy tuần tự cả ba model:

```powershell
.\scripts\training\retrain_models.ps1 -Model all -RunName cleaned_v1_seed42
```

## Theo dõi hoặc chạy lại

- `Ctrl+C` sẽ lưu `interrupt.pt`.
- Để resume đúng một model, dùng trực tiếp `train_unet.py`,
  `train_segformer.py` hoặc `train_vmamba.py` với `--resume` trỏ đến checkpoint
  đó và cùng `--output-dir`, `--run-name`, dataset/split.
- Không dùng lại cùng `-RunName` cho một run mới; script chặn việc đó để không
  ghi đè lịch sử, checkpoint hoặc metrics cũ.
