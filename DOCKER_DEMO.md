# 3CAD-ANI Docker Demo

Gói này chạy inference thật trên ảnh tải lên bằng U-Net/ResNet18 và
SegFormer-B0. Chế độ U-Net + SegFormer áp dụng spatial policy đã được khóa từ
Validation. VMamba vẫn có trong báo cáo kết quả nhưng không nằm trong runtime
mặc định vì cần CUDA selective-scan tương thích riêng.

## Yêu cầu

- Docker Desktop đang chạy;
- tối thiểu 8 GB RAM trống được khuyến nghị;
- Internet trong lần build đầu tiên để tải image và dependency;
- khoảng 8 GB dung lượng trống cho Docker image/build cache.

Không cần cài Python, Node.js, PyTorch hoặc CUDA trên máy nhận.

## Chạy

Mở PowerShell trong thư mục đã giải nén:

```powershell
.\start_docker_demo.ps1
```

Hoặc chạy trực tiếp:

```powershell
docker compose up -d --build
```

Mở `http://localhost:3000`. Lần inference đầu tiên chậm hơn vì backend mới nạp
checkpoint. Trên CPU, ảnh 1024x1024 có thể mất nhiều thời gian.

Các mode sẵn sàng trong bản portable:

- Original U-Net;
- Original SegFormer;
- U-Net Adaptive;
- SegFormer Adaptive;
- U-Net + SegFormer Spatial.

Các mode chứa VMamba được hiển thị là unavailable; đây là trạng thái có chủ ý,
không phải lỗi cài đặt.

## Dừng và xử lý lỗi

```powershell
.\stop_docker_demo.ps1
docker compose logs -f backend
docker compose logs -f frontend
```

Health check: `http://localhost:8000/health`.

Nếu port 3000 hoặc 8000 đang được dùng, sửa phần `ports` trong `compose.yaml`.

## Sample và kết quả

- Ảnh thử: `sample_images/input/`;
- GT tham khảo: `sample_images/ground_truth/`;
- visualization của ba baseline: `sample_images/reference_outputs/`;
- dashboard kết quả cuối: `artifacts/reports/final/visualizations/index.html`;
- bảng luận văn: `artifacts/reports/final/thesis_evaluation_report/tables/`;
- bảng decision policy: `artifacts/reports/final/decision_and_test_audit/tables/`.

Ảnh mẫu trích từ 3CAD ANI chỉ dành cho việc đánh giá riêng. Không tái phân phối
công khai khi chưa xác nhận điều khoản dataset. Xem `docs/DATASET_PROVENANCE.md`.

