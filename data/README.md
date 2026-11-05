# Dữ liệu ống hút cục bộ

Ảnh gốc nằm trong `straw_defects_raw/`; split và mask được sinh bằng công cụ
trong `scripts/data/` hoặc protocol nghiên cứu. Xem `configs/project.yaml`,
`configs/straw_thesis.json` và `docs/STRAW_THESIS.md` cho từng pipeline.
Dữ liệu ảnh, mask và cache không được commit. Hồ sơ nguồn/quyền sử dụng cần
xác nhận trong `HANDOVER_STATUS.md`.

Dùng `val_degraded_v2/` và `test_degraded_v2/` cho ảnh suy giảm. Bộ cũ
`val_degraded/` và `test_degraded/` có thể chứa ảnh bị ghi đè do tên file
không phân biệt thương hiệu; không dùng manifest cũ để hiệu chuẩn/báo cáo.
