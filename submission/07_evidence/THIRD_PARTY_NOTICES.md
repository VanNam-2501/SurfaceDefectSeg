# Nguồn bên thứ ba và việc cần hoàn thiện về bản quyền

Tài liệu này là inventory kỹ thuật, không phải tư vấn pháp lý.

## VMamba

- Upstream: `https://github.com/MzeroMiko/VMamba.git`.
- Commit đang có: `2ed52ead062a51a64521ed3871d52914bf532876`.
- License có trong vendor tree: MIT, Copyright © 2024 MzeroMiko.
- File runtime giữ trong dự án: `src/threecad_segmentation/third_party/VMamba/vmamba.py`.
- Thay đổi: thêm 15 dòng fallback decorator/type khi Triton không khả dụng;
  đường chạy không Triton dùng implementation khác và không thực thi kernel giả.
- Các thư mục nghiên cứu upstream không cần cho runtime đã được chuyển vào
  `archive/review_candidates/2026-08-21/vmamba_upstream_optional/`.

Khi nộp source phải giữ file LICENSE upstream và ghi rõ patch local.

## Model/dependency chính cần đưa vào dependency notice cuối

- PyTorch/Torchvision và pretrained ResNet18.
- Scikit-learn metrics cho đánh giá (không dùng learned verifier).
- Hugging Face Transformers và `nvidia/mit-b0`.
- OpenCV, NumPy, pandas, Pillow, Matplotlib.
- FastAPI/Uvicorn.
- React/Vinext/Vite và các package trong `web_demo/package-lock.json`.
- CUDA/selective-scan wheel dùng cho VMamba trên Kaggle.

Trước khi công bố hoặc nộp công khai, sinh viên phải tạo danh sách phiên bản và
license từ environment thật, giữ notice/license theo yêu cầu của từng package,
và kiểm tra điều khoản của pretrained weights.

## Dataset 3CAD

- Dataset: **3CAD: A Large-Scale Real-World 3C Product Dataset for
  Unsupervised Anomaly Detection**.
- Phần dùng trong đồ án: **Aluminum New Ipad (ANI)**, tên thư mục cục bộ
  `Aluminum_New_Ipad`.
- Bài báo chính thức: https://ojs.aaai.org/index.php/AAAI/article/view/32993
- Repository chính thức: https://github.com/ShuCvlab/3CAD
- Hồ sơ đối chiếu dữ liệu cục bộ: `DATASET_PROVENANCE.md`.

Tại thời điểm kiểm tra, bài báo và README repository cho biết nguồn/cấu trúc
dữ liệu nhưng chưa cung cấp license dữ liệu rõ ràng. Không đồng nghĩa
"repository công khai" với quyền tái phân phối. Trước khi upload dataset hoặc
ảnh sản xuất ra công khai, phải lưu điều khoản tải dữ liệu hay xác nhận quyền
sử dụng/công bố từ nguồn cung cấp.
