# Gói code Kaggle: 21 thí nghiệm, có lưu và tiếp tục

`straw_kaggle_code.zip` chỉ chứa code, configs, notebook START_HERE, kiểm thử,
tài liệu và CSV split cố định. **Không chứa dataset, ảnh, checkpoint hoặc weight.**
Dataset ảnh gốc được gắn trực tiếp từ Kaggle bằng Add Input.

**Cách gọn nhất:** import trực tiếp `STRAW_KAGGLE_ALL_EXPERIMENTS.ipynb`.
Notebook này nhúng toàn bộ ZIP code (có SHA-256), chỉ cần Add Input dataset ảnh
gốc. Không cần upload thêm ZIP code. Mọi tùy chọn và cơ chế resume bên dưới
áp dụng như nhau. ZIP source được giao riêng để đọc mã thuận tiện.

## Chạy lần đầu

1. Upload ZIP code thành Kaggle Dataset riêng tư, hoặc giải nén rồi upload các
   file bên trong. Add Input dataset code và dataset ảnh gốc vào notebook.
2. Import `START_HERE.ipynb` từ ZIP; hoặc tạo notebook mới rồi dán nội dung
   `KAGGLE_RUN_CELL.txt` được giao bên cạnh ZIP vào một cell.
3. Bật GPU và Internet để tải weight ResNet18 chính thức (~45 MB). Giữ cặp
   torch/torchvision có sẵn của Kaggle. Nếu cần chạy offline, gắn file
   `resnet18-f37072fd.pth` làm Input; tự tìm hoặc đặt `PRETRAINED_FILE`.
4. Giữ `RAW_ROOT = None` để tự tìm ảnh, sau đó chọn
   **Save Version → Save & Run All**.

Notebook mặc định chạy B0–B6 × seed 42/43/44 = **21 lượt nối tiếp**, tối đa
40 epoch/lượt, patience 10. Không train 21 mô hình đồng thời; dùng một GPU.
Hiện chưa tự tận dụng GPU T4 thứ hai. Chưa bảo đảm đủ 21 lượt trong một phiên.

## Đường dẫn dataset

Runner dò `/kaggle/input` theo cấu trúc nhãn/thương hiệu/tên ảnh được ghi trong
CSV; không hard-code tên tài khoản, slug dataset hay độ sâu mount của Kaggle.
Hỗ trợ cả cấu trúc có và không có thư mục ngoài `Straw Defects in Juice Box`.
Phải có đủ 2.000 đường dẫn ảnh gốc; cache kiểm tra SHA-256 và kích thước để
không vô tình dùng bản đã resize hoặc dataset khác. Nếu không tìm thấy hoặc
có nhiều bản phù hợp, chương trình báo lỗi rõ ràng và yêu cầu đặt `RAW_ROOT`.

Ví dụ (đây là ví dụ, không phải đường dẫn mặc định được giả định):

```python
RAW_ROOT = "/kaggle/input/juice-box-straw-defects/Straw Defects in Juice Box"
```

`RAW_ROOT` có thể trỏ tới thư mục đó hoặc thư mục cha. Chương trình đọc ảnh
trực tiếp từ Input, tạo symlink tạm trong working và ghi alias ổn định vào CSV.
Khi chuyển phiên, alias trỏ tới mount mới nhưng fingerprint CSV không thay đổi.
Không copy ảnh gốc vào working hoặc resume ZIP. Link tạm được gỡ trước khi
Kaggle lưu Output để tránh lưu nhầm dataset.

## Lưu và tiếp tục khi hết giờ

Sau mỗi epoch lưu `last.pt` gồm model, optimizer, scheduler, AMP scaler, RNG,
lịch sử, best score và early stopping. `best.pt` chọn bằng Validation.
LR schedule vẫn theo 40 epoch, không bắt đầu chu kỳ mới sau khi resume.

`SESSION_HOURS = 10.5` tính từ lúc cell bắt đầu, gồm chuẩn bị dữ liệu/cache.
Trước mỗi epoch, nếu còn ít hơn max(10 phút, 2 × epoch lâu nhất của lượt đó),
chương trình dừng bình thường, sinh báo cáo tiến độ và `straw_resume.zip`.
Kaggle sau đó lưu Output của phiên. Khoảng dự phòng giảm nguy cơ bị ngắt giữa
chừng; không thể bảo đảm trước lỗi hệ thống hoặc epoch kéo dài bất thường.

Phiên tiếp theo:

1. Giữ nguyên input code và dataset ảnh.
2. **Add Input → Notebook Output** của phiên trước; cần có `straw_resume.zip`.
   Hoặc gắn file resume ZIP do chính bạn tạo. Runner cũng đọc được ZIP đã giải
   nén nếu Kaggle tự giải nén khi upload thủ công.
3. Chạy lại notebook. `RESUME_FROM = None` tự tìm đúng một file resume.
   Nếu có nhiều file, điền đường dẫn cụ thể; không tự chọn một bản bằng phỏng đoán.
4. Lượt đã hoàn tất được kiểm tra hash rồi bỏ qua. Lượt dang dở tiếp tục ở epoch
   kế tiếp; các lượt còn lại tự chạy tiếp.

Không thay code/config/bundle/PROJECT_ROOT giữa các phiên của cùng thí nghiệm.
Mặc định PROJECT_ROOT là `/kaggle/working/straw-global-local`.
Resume ZIP chứa protocol, cache view (~1,1 GB), checkpoint, predictions,
report, log và manifest kiểm tra hash. Không có ảnh gốc hoặc weight pretrained.

Nếu chạy lại trong cùng phiên, tiến độ đang có trong working được ưu tiên,
không bị ghi đè bằng archive cũ. Khi Kaggle đổi GPU hoặc thư viện giữa các
phiên, chương trình ghi nhận thay đổi; tính bit-for-bit chỉ được kiểm chứng
trong cùng môi trường. Không dùng checkpoint của người lạ.

**Giới hạn:** checkpoint trên ổ tạm chỉ tồn tại qua phiên mới nếu Kaggle đã
lưu Output hoặc bạn đã tải file. Nếu phiên bị cưỡng bức ngắt trước khi lưu,
có thể mất tiến độ của phiên đó. Chương trình không tự upload checkpoint lên
cloud theo thời gian thực và không sử dụng token/tài khoản của bạn.

## Kết quả và đánh giá Test

Trong `artifacts/straw_global_local/confirmatory_kaggle_v1`:

- `plan.json`: danh sách đủ 21 lượt.
- `runs/*/last.pt`, `best.pt`, `history.json`, `val_predictions.npz`.
- `report/validation_runs.csv`: kết quả Validation của các lượt đã hoàn tất.
- `runs/*/benchmark.json`: chi phí inference trên ảnh Validation gốc, một seed
  mỗi cấu hình, sau khi train đủ. Nếu hết giờ trước benchmark, lần sau chạy tiếp.
- `freeze.json`, `test_predictions.npz`, `test.json`, báo cáo so sánh/CI: chỉ
  sinh khi đã bật Test và có review thực tế.

Mặc định `EVALUATE_TEST = False`. Muốn đánh giá Test sau khi hoàn tất rà dữ liệu
và related work, đặt `EVALUATE_TEST = True`, cung cấp `REVIEW_EVIDENCE` gồm
`labels_reviewed`, `related_groups_reviewed`, `dataset_rights_recorded`,
`related_work_reviewed`, `reviewer`, `notes` cùng bằng chứng thật.
Không đánh dấu True chỉ để vượt điều kiện. Chạy lại sẽ bỏ qua training đã
xong, khóa lựa chọn bằng Validation rồi đánh giá Test. Không dùng Test để tuning.

Trạng thái tổng ở `/kaggle/working/straw_session_status.json`, có số lượt xong,
lịch sử phiên, GPU/thư viện, lỗi nếu có và bước tiếp theo. Không cam kết phương
pháp mới thắng baseline. Dataset đã được dùng trong phát triển, không gọi Test
là một bộ xác nhận độc lập mới.

## Kiểm thử và đóng gói lại

```text
python -m unittest discover -s tests -p test_global_local.py -v
python -m unittest discover -s tests -p test_kaggle_session.py -v
python scripts/research/package_kaggle_all_in_one.py
```

Nội dung ZIP đổi sẽ có fingerprint mới. Giữ đúng ZIP ban đầu cho mọi phiên
của cùng bộ 21 lượt. Kiểm thử tại máy phát triển không đồng nghĩa notebook đã
được thực thi trên Kaggle.

Giới hạn phiên và cách lưu: https://www.kaggle.com/docs/notebooks
