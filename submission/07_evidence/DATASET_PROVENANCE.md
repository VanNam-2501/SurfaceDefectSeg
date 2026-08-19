# Nguồn gốc dataset và đối chiếu bản cục bộ

Ngày xác minh: **2026-08-19**.

## Nhận diện dataset

- Tên chính thức: **3CAD: A Large-Scale Real-World 3C Product Dataset for
  Unsupervised Anomaly Detection**.
- Công bố: AAAI 2025.
- Nhóm dùng trong đồ án: **Aluminum New Ipad**, viết tắt **ANI**.
- Tên thư mục cục bộ: `data/3cad_ani`.
- Bài báo chính thức:
  https://ojs.aaai.org/index.php/AAAI/article/view/32993
- Bản arXiv: https://arxiv.org/abs/2502.05761
- Repository chính thức: https://github.com/ShuCvlab/3CAD

Bài báo và repository mô tả 3CAD gồm 8 nhóm sản phẩm, 27.039 ảnh có nhãn bất
thường mức pixel. Riêng ANI có 2.233 ảnh train và 4.936 ảnh test, tổng 7.169.

## Đối chiếu dữ liệu trong workspace

| Mốc | Số ảnh | Bằng chứng |
|---|---:|---|
| Audit nguồn cục bộ | 7.169 | `data/3cad_ani/dataset_audit/REPORT.md` |
| Manifest nguồn | 7.169 | `data/3cad_ani/dataset_audit/manifest.csv` |
| Frozen Train | 5.733 | `data/3cad_ani/dataset_audit/splits/train.csv` |
| Frozen Validation | 718 | `data/3cad_ani/dataset_audit/splits/val.csv` |
| Frozen Test | 717 | `data/3cad_ani/dataset_audit/splits/test.csv` |
| Tổng dùng trong thí nghiệm | 7.168 | Tổng ba frozen split |

Chênh lệch một ảnh đã được ghi tại
`data/3cad_ani/dataset_audit/cleaning_issues.csv`: ảnh
`test/bump/001541.png` bị loại vì file nguồn không còn và audit trước đó xác
định nó trùng SHA-256 với `test/bump/001540.png`. Vì vậy báo cáo phải phân biệt
rõ **7.169 ảnh ANI trong bản nguồn đã audit** với **7.168 ảnh thật sự dùng cho
thí nghiệm**.

## Trạng thái quyền sử dụng

Tên, bài báo và repository của dataset đã được xác định. Tuy nhiên, tại thời
điểm xác minh, README repository chính thức không nêu license dữ liệu rõ ràng.
Các việc còn phải làm trước khi công bố dataset/ảnh:

1. Lưu lại trang hoặc điều khoản tại nơi đã tải dataset.
2. Nếu điều khoản không rõ, xin xác nhận quyền dùng cho nghiên cứu và quyền tái
   phân phối từ bên cung cấp.
3. Chỉ nộp mã, manifest và hướng dẫn tải nếu không có quyền phát hành lại ảnh.
4. Trích dẫn bài báo 3CAD trong báo cáo và slide.

## Trích dẫn đề xuất

```bibtex
@inproceedings{yang2025threecad,
  title={3CAD: A Large-Scale Real-World 3C Product Dataset for Unsupervised Anomaly Detection},
  author={Yang, Enquan and Xing, Peng and Sun, Hanyang and Guo, Wenbo and Ma, Yuanwei and Li, Zechao and Zeng, Dan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
