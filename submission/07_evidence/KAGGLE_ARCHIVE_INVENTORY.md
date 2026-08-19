# Kiểm kê archive Kaggle sau cleanup

Ngày 2026-08-19, toàn bộ entry của từng archive cũ đã được so sánh với archive
cuối. Archive cuối chứa mọi đường dẫn của các bản cũ và đọc/decompress thành
công toàn bộ 12.934 entry. Vì vậy chỉ giữ gói cuối trong workspace; SHA-256 của
các bản đã chuyển sang quarantine vẫn được lưu để truy vết và có thể phục hồi.

| File | Dung lượng | SHA256 | Trạng thái đề xuất |
|---|---:|---|---|
| `TTTN_Kaggle_Project_Data_20260816.zip` | 477,6 MB | `069A34FFE5913E85AA74C7F0897C0B4748F9D87F8CFAA0FD194B2BF92ECA42E5` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_All3_Pipeline_20260818.zip` | 490,9 MB | `5F8D7FFF61E5228F1DB697ED5F1F50801245B42D8E09A9AF5EDF32329B448A16` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_All3_Evaluation_Complete_20260818.zip` | 1.011,7 MB | `76F949F2DCC961D0CE1059AA4B1A7221A9EFA21D5FCB13333D2155F0D706E660` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_All3_Evaluation_Audit_20260818.zip` | 1.059,5 MB | `B389E3CFA1D4A06F9A20B697DD7EF8473BED1B0C4894D7DC021B7099A9AEF16E` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_All3_Evaluation_Audit_Final_20260818.zip` | 1.059,5 MB | `DDB36A4F58FF11AAA63943F0AF208C2B218E578432F0A3A03778C95869403455` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_All3_Evaluation_DataAudit_Final_20260818.zip` | 1.059,5 MB | `658CF45B9B0AD0A6370E293DF96D5ABF7B841502FC85F79B91A098A35E386A9A` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_All3_Evaluation_DataAudit_Final_v2_20260818.zip` | 1.059,5 MB | `937016DFEC633A1AD98F75937ADA8894774369A19827DAA39144C631903B4188` | QUARANTINE — bị bản cuối thay thế |
| `TTTN_Kaggle_Selected_Thesis_Experiments_Final_20260818.zip` | 1.059,5 MB | `C8B55B63C76FBF553DD4AB1AC73B1EF6A59B4A21BDA4E868ED2FBAC3F0C8DBB6` | GIỮ — archive cuối duy nhất |

Chi tiết phương pháp kiểm tra và các bản sao khác đã loại nằm trong
`CLEANUP_REPORT.md`.
