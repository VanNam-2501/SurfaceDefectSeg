"""Summarize an already completed local B1 check."""
import json
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/local_b1_check'

def main():
    for seed in (42,43,44):
        folder = OUT / f'B1_s{seed}'
        marker = json.loads((folder / 'complete.json').read_text())
        assert hashlib.sha256((folder / 'best.pt').read_bytes()).hexdigest() == marker['best_checkpoint']
    summary = pd.read_csv(OUT / 'test_summary.csv')
    bench = pd.read_csv(OUT / 'benchmarks.csv')
    errors = pd.concat([pd.read_csv(OUT / f'B1_s{s}_errors.csv').assign(seed=s) for s in (42,43,44)])
    unique = errors.drop_duplicates('sample_id')
    if len(unique):
        sheet = Image.new('RGB',(400*len(unique),450),'white')
        draw = ImageDraw.Draw(sheet)
        for i,row in enumerate(unique.itertuples()):
            with Image.open(ROOT / 'data/straw_defects_raw' / row.relative_path) as im:
                thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(380,380))
            sheet.paste(thumb,(i*400+(400-thumb.width)//2,0))
            draw.text((i*400+10,390),f'True: {row.class_name} -> Pred: {row.prediction}',fill='black')
            draw.text((i*400+10,415),str(row.sample_id),fill='black')
        sheet.save(OUT / 'errors.jpg')
    lines = ['# Kiểm tra B1 trên laptop', '',
        'Phần cứng: Ryzen 7 5800H, RTX 3050 Laptop 4 GB, RAM 23.35 GiB. CPU dùng 4 luồng.',
        'Checkpoint Kaggle được đối chiếu SHA-256; split và ảnh test được kiểm tra với manifest gốc.',
        'Không huấn luyện lại; không chọn ngưỡng dựa trên test. Đây là đo trên laptop, chưa phải thiết bị biên chuyên dụng.', '',
        '## Chất lượng trên 299 ảnh test (241 lỗi, 58 tốt)', '',
        '| Seed | Precision | Macro-F1 (%) | Accuracy (%) | FN | FP | Sai loại lỗi |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.seed} | {r.precision} | {100*r.macro_f1:.4f} | {100*r.accuracy:.4f} | {r.fn} | {r.fp} | {r.wrong_defect_type} |')
    lines += ['', '## Tốc độ batch=1', '',
        '100 lượt đo sau 10 lượt khởi động, dùng 20 ảnh validation. End-to-end gồm đọc/giải mã ảnh, resize 512, chuẩn hóa, chuyển thiết bị và forward; chưa gồm camera hoặc giao diện.', '',
        '| Thiết bị | Precision | Forward p50 (ms) | End-to-end p50 (ms) | End-to-end p95 (ms) |',
        '|---|---|---:|---:|---:|']
    for r in bench.itertuples():
        lines.append(f'| {r.device} | {r.precision} | {r.p50_forward_ms:.2f} | {r.p50_end_to_end_ms:.2f} | {r.p95_end_to_end_ms:.2f} |')
    lines += ['', '## Các ảnh dự đoán sai', '']
    for r in unique.itertuples():
        lines.append(f'- {r.sample_id}: {r.class_name} → {r.prediction}; `{r.relative_path}`')
    lines += ['', '## Giới hạn', '',
        'Ba seed dùng chung tập test. FP=0 trên 58 ảnh tốt không chứng minh FPR thực tế bằng 0. Chưa kiểm tra ảnh ngoài bộ dữ liệu, camera trực tiếp hoặc INT8.',
        'FP16 được kiểm tra với seed 42; chưa kiểm tra seed 43/44. RSS trong bảng CSV là của toàn tiến trình đo với nhiều mô hình, không phải bộ nhớ triển khai độc lập.',
        'validation_threshold_survey.csv chỉ là khảo sát trên validation; không tự động thay đổi ngưỡng ứng dụng.', '',
        'Chạy lại: `.venv\\Scripts\\python.exe scripts/research/local_b1_check.py`, sau đó chạy `scripts/research/local_b1_report.py` bằng cùng Python.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report saved: ' + str(OUT / 'REPORT.md'))

if __name__ == '__main__': main()
