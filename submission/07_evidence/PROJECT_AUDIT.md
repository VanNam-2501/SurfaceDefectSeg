# Audit cấu trúc dự án trước migration

> Đây là ảnh chụp cấu trúc cũ trước lần chuẩn hóa 2026-08-19. Xem
> `PROJECT_AUDIT_CURRENT.md` và `../../docs/MIGRATION_REPORT.md` để biết trạng
> thái hiện hành.

Ngày kiểm tra: 2026-08-19.

## Kết luận ngắn

Mã nguồn và kết quả kỹ thuật đã đủ để tiếp tục viết báo cáo, nhưng bộ hồ sơ
nộp chưa hoàn chỉnh. Thiếu lớn nhất là Git history, báo cáo, slide, video, nhật
ký thực tập và xác nhận đơn vị. AI log mới ở mức hồi cứu và chưa có commit.

## Dung lượng theo khu vực

| Khu vực | Số file xấp xỉ | Dung lượng | Nhận xét |
|---|---:|---:|---|
| `cleanup_quarantine/` | 12.305 | 7,81 GB | Bản cũ/bản lặp đã tách khỏi cây làm việc; chờ duyệt xóa |
| `kaggle_upload/` | 6 | 1,61 GB | 3 checkpoint, VMamba wheel và một ZIP Kaggle cuối |
| `Aluminum_New_Ipad/` | 18.342 | 0,76 GB | Dataset/audit; không commit Git thường |
| Canonical ML bundle | 1.848 | 0,03 GB | Source và vendor VMamba; dataset dùng junction tới nguồn chuẩn |
| `web_demo/` | 21.103 | 0,39 GB | Phần lớn là `node_modules`; output sinh tự động đã quarantine |
| `final_thesis_deliverables/` | 5.027 | 0,22 GB | Bảng, gallery, audit và visualizations |

## Nguồn chuẩn đã xác định

- ML: `Aluminum_Surface_Defect_Segmentation_Bundle/Aluminum_Surface_Defect_Segmentation/`.
- Data/audit: `Aluminum_New_Ipad/` — nhóm Aluminum New Ipad (ANI) của 3CAD.
- Demo: `web_demo/`.
- Review tool: `dataset_review_tool/`.
- Trọng số cuối: `kaggle_upload/all3_eval_weights_20260818/`.
- Kết quả cuối: `final_thesis_deliverables/`.

## Vấn đề phát hiện

### P0 — bắt buộc trước khi nộp

1. Root chưa có Git repository; repo rỗng trước đây của web demo đã chuyển vào
   quarantine để chuẩn bị dùng một Git repository duy nhất ở root.
2. Chưa có report/slide/video/internship log/company confirmation.
3. AI log chưa có exact model/build, raw export và commit.
4. Nguồn dataset đã xác định là 3CAD/ANI và được ghi tại
   `DATASET_PROVENANCE.md`; vẫn thiếu license/điều khoản phân phối dữ liệu rõ
   ràng hoặc xác nhận quyền công bố từ nguồn cung cấp.

### P1 — ảnh hưởng tính chính xác hồ sơ

1. Protocol cũ ghi effective batch = 4, không khớp checkpoint cuối.
2. Training history/config/environment độc lập của ba run cuối không có trong
   workspace; config vẫn còn bên trong checkpoint, nhưng E6 chưa đủ.
3. `third_party/VMamba/vmamba.py` có 15 dòng sửa local chưa có commit/patch
   chính thức trong repository dự án.
4. Learned Verifier và Learned Hybrid đã được chuyển vào archive/learned_verifier và không còn thuộc pipeline mặc định.

### P2 — vệ sinh và dung lượng

1. Quarantine còn 7,81 GB. Đây là nơi duy nhất chứa ZIP cũ, bản dataset lặp,
   lần chạy VMamba dở và cache; có thể xóa sau khi người dùng duyệt.
2. `.venv` 5,03 GB và `web_demo/node_modules` 0,39 GB là runtime tái tạo được
   nhưng được giữ để demo/review chạy ngay.
3. File near-duplicate audit 221 MB được giữ vì là bằng chứng kiểm tra dữ liệu,
   không phải cache ứng dụng.

## Cleanup đã thực hiện

- Xác định và ghi rõ canonical source/result paths.
- Bổ sung `.gitignore` cho dataset, checkpoint, ZIP, cache và workspace sinh tự động.
- Bổ sung cấu hình checkpoint thực tế.
- Chuyển test learned verifier sang `unittest`, không bắt buộc cài pytest.
- Launcher demo không còn mặc định trỏ tới `E:\Dowload`; dùng checkpoint và
  final U-Net+VMamba policy trong workspace.
- Tạo skeleton hồ sơ nộp, AI log, verification report và artifact manifests.
- Chỉ giữ một ZIP Kaggle cuối trong cây hoạt động; chuyển 7 bản cũ vào
  quarantine sau khi đối chiếu toàn bộ entry.
- Chuyển checkpoint lặp, output VMamba dở, wheel lặp, cache Python/web và Git
  rỗng của web demo vào quarantine. `web_demo/build/sites-vite-plugin.ts` được
  phục hồi vì kiểm thử chứng minh đây là mã nguồn, không phải cache.
- Thay dataset lặp trong ML bundle bằng junction tới `Aluminum_New_Ipad/`;
  bản sao cũ vẫn nằm trong quarantine nên có thể phục hồi.

## Cleanup chưa thực hiện có chủ ý

- Không xóa vĩnh viễn quarantine; chờ người dùng kiểm tra và phê duyệt.
- Không xóa `.venv`/`node_modules` để tránh làm gián đoạn demo.
- Không khởi tạo hoặc commit Git thay sinh viên vì cần quyết định repository,
  danh tính commit và cam kết lịch sử trung thực.
- Không tự tạo nội dung báo cáo, nhật ký thực tập hoặc xác nhận đơn vị không có
  bằng chứng thực tế.
