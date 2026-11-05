# Thành phần bên thứ ba của bản bàn giao ống hút

| Thành phần | Vai trò / nơi ghi phiên bản |
|---|---|
| NumPy, Pillow | Viewer Windows; khóa tại `requirements/deepstream-demo.txt` |
| NVIDIA DeepStream / TensorRT / CUDA | Runtime GPU trong image NVIDIA; `engine_cache.json`, `sdk_version.txt` |
| PyTorch / Torchvision / pretrained ResNet18 | Huấn luyện, export; environment trong hồ sơ thí nghiệm |
| VMamba (tùy chọn) | Mã và license gốc tại `src/straw_defect_cls/third_party/VMamba/`; classifier VMamba vẫn cần CUDA/selective-scan tương thích |
| pandas, SciPy, scikit-learn, Matplotlib, tqdm | Nghiên cứu/đánh giá; `requirements/straw-thesis.txt` |
| Docker, WSL, Ubuntu, GStreamer, FFmpeg | Môi trường, decode và replay; cài riêng bằng runbook |

ZIP đầy đủ kèm hai wheel NumPy/Pillow nguyên bản, giữ nguyên license/notice
bên trong wheel, để cài offline trên Windows Python 3.12 x64. Gói không kèm
image SDK hoặc pretrained ResNet18 nguyên bản. Checkpoint đã huấn luyện và engine được kèm để bàn giao
nội bộ. Giữ license/notice đi kèm từng thành phần khi cài hoặc phân phối
runtime riêng. Dự án chưa chỉ định license mã nguồn công khai; không tự gán
MIT hoặc quyền phát hành dữ liệu trong gói bàn giao này.

Ảnh/clip mẫu lấy từ dataset ống hút hiện có. Hồ sơ nguồn và quyền sử dụng
chưa có xác nhận hoàn chỉnh; xem `HANDOVER_STATUS.md` ở gốc gói.
