# Runtime binaries

`wheels/demo/` chứa wheel NumPy/Pillow phục vụ bộ bàn giao offline; không đưa
binary vào Git. DeepStream/TensorRT và engine GPU được cung cấp qua bộ bàn
giao riêng, với kiểm tra môi trường trong `docs/DEEPSTREAM_HANDOVER.md`.

Thí nghiệm VMamba tùy chọn cần CUDA/selective-scan tương thích; mã bên thứ ba
và license nằm trong `src/straw_defect_cls/third_party/VMamba/`.
