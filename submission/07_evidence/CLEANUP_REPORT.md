# Báo cáo dọn dẹp workspace trước migration

> Các đường dẫn trong báo cáo này phản ánh thời điểm cleanup ban đầu. Cấu trúc
> hiện hành và ánh xạ đầy đủ nằm tại `../../docs/MIGRATION_MANIFEST.md`.

Ngày thực hiện: **2026-08-19**.

## Nguyên tắc

- Giữ một nguồn chuẩn cho mã, dữ liệu, checkpoint và kết quả cuối.
- Chỉ xóa bản sao đã đối chiếu, cache tái tạo được và lần chạy dở dang.
- Không xóa review state, mask đã sửa, frozen split, checkpoint cuối, bảng kết
  quả, gallery, notebook hoặc hồ sơ nộp.
- Ghi lại SHA-256 của archive bị loại để có dấu vết kiểm toán.

## Nguồn chuẩn sau cleanup

| Thành phần | Nguồn chuẩn |
|---|---|
| Dataset | `Aluminum_New_Ipad/` |
| Mã ML | `Aluminum_Surface_Defect_Segmentation_Bundle/Aluminum_Surface_Defect_Segmentation/` |
| Checkpoint cuối | `kaggle_upload/all3_eval_weights_20260818/` |
| VMamba wheel Kaggle | `kaggle_upload/mamba-wheel/` |
| Gói Kaggle cuối | `kaggle_upload/TTTN_Kaggle_Selected_Thesis_Experiments_Final_20260818.zip` |
| Kết quả và biểu đồ | `final_thesis_deliverables/` |
| Prediction cache Val/Test | `decision_workspace/predictions/` |
| Review data và tiến độ | `dataset_review_tool/` |
| Demo | `web_demo/` |

## Xác minh trước khi xóa

- Gói Kaggle cuối đọc/decompress thành công toàn bộ **12.934 entry**, tổng
  **1.171.953.312 byte** chưa nén.
- Mỗi trong 7 ZIP Kaggle cũ có **0 đường dẫn thiếu** trong gói cuối. Một số file
  cùng tên có kích thước khác vì gói cuối chứa phiên bản mã/notebook mới hơn.
- Bản dataset nằm trong ML bundle có 11.109 file; mọi đường dẫn đều có trong
  dataset gốc và cùng kích thước. Dataset gốc còn là bản đầy đủ hơn vì chứa
  manifest, audit và mask Good.
- `best/vmamba_best.pt` trùng SHA-256 với checkpoint VMamba cuối. Thư mục
  `best/vmamba_predictions` chỉ có 39 ảnh Validation của lần export dở dang.
- Hai bản VMamba wheel và hai file build-info trùng SHA-256.
- `web_demo/.git` không có ref, object hoặc commit; đây chỉ là repository rỗng.

## Đã loại khỏi cây làm việc chính

Các mục dưới đây được chuyển vào `cleanup_quarantine/2026-08-19/`, chưa bị xóa
vĩnh viễn.

### Archive Kaggle đã bị gói cuối thay thế

| File | SHA-256 |
|---|---|
| `TTTN_Kaggle_Project_Data_20260816.zip` | `069A34FFE5913E85AA74C7F0897C0B4748F9D87F8CFAA0FD194B2BF92ECA42E5` |
| `TTTN_Kaggle_All3_Pipeline_20260818.zip` | `5F8D7FFF61E5228F1DB697ED5F1F50801245B42D8E09A9AF5EDF32329B448A16` |
| `TTTN_Kaggle_All3_Evaluation_Complete_20260818.zip` | `76F949F2DCC961D0CE1059AA4B1A7221A9EFA21D5FCB13333D2155F0D706E660` |
| `TTTN_Kaggle_All3_Evaluation_Audit_20260818.zip` | `B389E3CFA1D4A06F9A20B697DD7EF8473BED1B0C4894D7DC021B7099A9AEF16E` |
| `TTTN_Kaggle_All3_Evaluation_Audit_Final_20260818.zip` | `DDB36A4F58FF11AAA63943F0AF208C2B218E578432F0A3A03778C95869403455` |
| `TTTN_Kaggle_All3_Evaluation_DataAudit_Final_20260818.zip` | `658CF45B9B0AD0A6370E293DF96D5ABF7B841502FC85F79B91A098A35E386A9A` |
| `TTTN_Kaggle_All3_Evaluation_DataAudit_Final_v2_20260818.zip` | `937016DFEC633A1AD98F75937ADA8894774369A19827DAA39144C631903B4188` |

### Bản sao và output dở dang

- `Aluminum_Surface_Defect_Segmentation_Bundle.zip` — archive source cũ,
  SHA-256 `D4CE564F00CA4D50A576AC4EF8313ACA32ADF65446A8F3C99787ACE2563C11C0`.
- `wheels-20260813T041359Z-1-001.zip` và thư mục bung tương ứng — trùng wheel
  đang giữ trong `kaggle_upload/mamba-wheel/`.
- `best/` — checkpoint trùng, nội dung checkpoint bị bung nhầm, export VMamba
  dở 39 ảnh và log lần chạy không hoàn tất.
- Dataset lặp trong ML bundle được thay bằng junction tới
  `E:\Project\TTTN\Aluminum_New_Ipad` để code cũ vẫn chạy bằng đường dẫn cũ.
- Cache Python ngoài `.venv`, các output web `.next`, `.vinext`, `.wrangler`,
  `dist` và log runtime cũ. `web_demo/build/sites-vite-plugin.ts` đã được xác
  định là mã nguồn bắt buộc và được phục hồi sau kiểm thử.
- `web_demo/.git` rỗng; giữ `third_party/VMamba/.git` để bảo toàn provenance
  upstream.

## Chủ động giữ lại

- `.venv/` và `web_demo/node_modules/`: lớn nhưng giúp demo/review chạy ngay;
  không đóng gói khi nộp.
- decision_workspace/ và adaptive_component_workspace/: output trung gian phục vụ tái kiểm tra.
- Learned Verifier/Hybrid cùng artifact cũ được bảo toàn trong archive/learned_verifier/.
- U-Net Val/Test prediction cache đã được gom từ kết quả ngoài workspace vào
  `decision_workspace/predictions/unet/`; các script thí nghiệm không còn mặc
  định phụ thuộc `E:\Dowload`.
- `Aluminum_New_Ipad/dataset_audit/near_duplicate_candidates_dhash_le4.csv`:
  lớn nhưng là bằng chứng audit dữ liệu; chỉ loại khỏi source release, không xóa.
- Gói Kaggle cuối: giữ làm bản tái lập ngoại tuyến; không đưa vào Git.

## Khả năng phục hồi

Mọi mục đã dọn vẫn nằm trong `cleanup_quarantine/2026-08-19/` và có thể chuyển
về vị trí cũ. Nội dung cần thiết cũng có trong gói Kaggle cuối, checkpoint chuẩn
hoặc có thể tái tạo từ mã. Cache web/Python có thể sinh lại. Junction dữ liệu
có thể thay bằng bản sao đã quarantine nếu cần tạo bundle độc lập.
