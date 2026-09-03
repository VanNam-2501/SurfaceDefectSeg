# SurfaceDefectSeg

Hệ thống phân đoạn khuyết tật bề mặt nhôm trên tập dữ liệu 3CAD-ANI, phục vụ huấn luyện, đánh giá mô hình, hậu xử lý theo thành phần liên thông và trực quan hóa kết quả.

## Tổng quan

Dự án triển khai ba kiến trúc phân đoạn:

| Mô hình | Kiến trúc | Trạng thái demo |
| --- | --- | --- |
| U-Net | ResNet-18 encoder | Hỗ trợ mặc định |
| SegFormer | MiT-B0 | Hỗ trợ mặc định |
| VMamba | VMamba-T | Yêu cầu CUDA và selective-scan tương thích |

Kết quả từ mô hình được xử lý theo ba nhóm chế độ:

- **Original:** sử dụng ngưỡng đã khóa trên tập Validation.
- **Adaptive:** lọc và hiệu chỉnh từng mô hình bằng chính sách thành phần.
- **Spatial:** kết hợp không gian giữa hai mô hình bằng policy đã cố định.

Mọi threshold và policy được lựa chọn trên Validation. Tập Test chỉ được dùng để đánh giá cuối cùng.

## Luồng xử lý

```text
Ảnh đầu vào
    -> mô hình phân đoạn
    -> probability mask
    -> threshold đã khóa
    -> adaptive/spatial policy
    -> mask dự đoán, overlay và thông tin thành phần
```

## Chạy demo bằng Docker

### Yêu cầu

- Docker Desktop và Docker Compose.
- Các checkpoint và decision policy trong `artifacts/`.
- Khoảng 8 GB RAM và 8 GB dung lượng trống cho quá trình build.

Khởi động:

```powershell
.\start_docker_demo.ps1
```

Hoặc:

```powershell
docker compose up -d --build
```

Truy cập:

- Giao diện: [http://localhost:3000](http://localhost:3000)
- API health check: [http://localhost:8000/health](http://localhost:8000/health)

Dừng hệ thống:

```powershell
.\stop_docker_demo.ps1
```

Docker mặc định chạy inference U-Net và SegFormer. VMamba không nằm trong runtime Docker mặc định vì phụ thuộc CUDA selective-scan theo môi trường. Xem [DOCKER_DEMO.md](DOCKER_DEMO.md) để biết cấu hình chi tiết.

## Cài đặt cục bộ

Yêu cầu Python 3.10 trở lên và Node.js.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/dev.txt

Set-Location apps/web_demo
npm install
Set-Location ../..
```

Khởi động demo:

```powershell
.\run_demo.ps1
```

Demo cục bộ chỉ kích hoạt mô hình khi checkpoint, policy và runtime tương ứng đều khả dụng. Chi tiết biến môi trường và đường dẫn checkpoint nằm trong [apps/web_demo/INFERENCE_SETUP.md](apps/web_demo/INFERENCE_SETUP.md).

## Thí nghiệm

Chạy lại pipeline quyết định từ probability cache:

```powershell
.\scripts\experiments\run_three_model_experiments.ps1
```

Train lại một mô hình:

```powershell
.\scripts\training\retrain_models.ps1 -Model unet -RunName unet_seed42_v2
```

Tài liệu phương pháp:

- [Experiment protocol](docs/experiments/protocol.md)
- [Final configuration](docs/experiments/final_config.md)
- [Decision pipeline](docs/decision_pipeline.md)

## Kiểm thử

Trên workspace đầy đủ dataset, checkpoint và artifact:

```powershell
.\verify.ps1 -IncludeWeb
```

Quy trình kiểm tra bao gồm tính hợp lệ của data split, decision policy, trạng thái huấn luyện, công cụ review dữ liệu và frontend.

## Cấu trúc dự án

```text
apps/                       Dataset Review và web demo
src/threecad_segmentation/  Model, evaluation và decision policy
scripts/                    Data, training, experiment, reporting, verification
tests/                      Kiểm thử mã ML và ứng dụng
experiments/                Notebook audit và pipeline Kaggle
docs/                       Protocol và tài liệu kỹ thuật
submission/                 Báo cáo và bằng chứng kiểm chứng
docker/                     Dockerfile cho backend và frontend
archive/                    Mã không còn thuộc luồng chạy chính
```

Mô tả đầy đủ nằm trong [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Dữ liệu và kết quả

Dataset, checkpoint, probability cache, báo cáo sinh tự động và các gói bàn giao có dung lượng lớn không được lưu trong Git.

- Hướng dẫn dữ liệu: [data/README.md](data/README.md)
- Nguồn gốc dataset: [submission/07_evidence/DATASET_PROVENANCE.md](submission/07_evidence/DATASET_PROVENANCE.md)
- Thông báo thành phần bên thứ ba: [submission/07_evidence/THIRD_PARTY_NOTICES.md](submission/07_evidence/THIRD_PARTY_NOTICES.md)

Để chạy đầy đủ trên máy khác, sử dụng source code cùng gói artifact/checkpoint được cung cấp riêng.
