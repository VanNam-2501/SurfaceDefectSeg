# Local data

`3cad_ani/` là tập con ANI (Aluminum New iPad) của 3CAD, gồm ảnh, mask và frozen
splits trong `dataset_audit/splits`. Dữ liệu lớn không được commit vào Git. Xem
`submission/07_evidence/DATASET_PROVENANCE.md` trước khi phân phối dữ liệu.

Với bộ ảnh ống hút, dùng `val_degraded_v2/` và `test_degraded_v2/` cho ảnh suy
giảm. Bộ cũ `val_degraded/` và `test_degraded/` có thể chứa ảnh bị ghi đè vì
tên file trước đây không phân biệt thương hiệu; không dùng các manifest cũ để
hiệu chuẩn hoặc báo cáo metric.

