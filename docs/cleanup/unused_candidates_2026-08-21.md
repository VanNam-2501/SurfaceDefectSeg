# Cleanup candidates — 2026-08-21

Đợt dọn này chỉ đưa những thành phần không tham gia luồng chạy hiện tại sang
`archive/review_candidates/2026-08-21/`. Thư mục `archive` bị Git bỏ qua để mã
nguồn bàn giao gọn, nhưng các tệp vẫn còn trên máy để kiểm tra và phục hồi.

## Đã xóa khỏi dự án

Những tệp chỉ phục vụ Google Colab đã được xóa theo yêu cầu:

- `docs/guides/colab.md`;
- bốn notebook `experiments/notebooks/training/*_colab.ipynb`;
- `scripts/setup/setup_vmamba_colab.sh`.

Pipeline Kaggle, notebook data-audit và wheel Kaggle vẫn được giữ nguyên.

## Đã chuyển sang khu xem xét

### `web_demo_scaffold/`

Scaffold Cloudflare/OpenAI Sites, D1/Drizzle, worker, ví dụ database, helper xác
thực ChatGPT và ba SVG placeholder. Không tệp nào được frontend, FastAPI,
launcher hoặc test hiện tại import. `vite.config.ts` và `package.json` đã được
rút gọn cho web demo local.

### `vmamba_upstream_optional/`

Các thư mục nghiên cứu upstream `analyze`, `classification`, `detection`,
`kernels`, `segmentation`, `assets` và hai shell script upstream. Adapter của dự
án chỉ import `third_party/VMamba/vmamba.py`, vì vậy đã giữ lại đúng runtime,
`LICENSE`, `README.md`, `requirements.txt` và `.gitignore` tại vị trí cũ.

### `out_of_scope_experiments/`

`compare_seeds.py` thuộc E9 nhiều seed, trong khi E9 đã được chốt ngoài phạm vi
bàn giao cuối.

### `legacy_docs/`

Checklist dự án cũ bị trùng chức năng với `submission/CHECKLIST.md` và chứa các
mục thí nghiệm không còn thuộc phạm vi.

### `legacy_wrappers/`

Năm PowerShell wrapper từng phục vụ các lần chạy VMamba/adaptive riêng lẻ. Các
wrapper này không còn được launcher, notebook, tài liệu hay test tham chiếu;
chức năng còn cần đã có trong `export_probability_cache.py`,
`run_decision_pipeline.ps1` và `run_three_model_experiments.ps1`.

## Cách phục hồi

Sao chép hoặc di chuyển tệp từ nhóm tương ứng về đúng đường dẫn con được giữ
trong thư mục archive. Ví dụ:

```powershell
Move-Item `
  .\archive\review_candidates\2026-08-21\web_demo_scaffold\db `
  .\apps\web_demo\db
```

Sau khi phục hồi scaffold web, cần phục hồi đồng thời dependency/cấu hình từ
lịch sử Git trước cleanup. Không phục hồi riêng lẻ `db` rồi chạy production.
