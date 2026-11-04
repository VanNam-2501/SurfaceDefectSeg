"""Append measured web evidence to REPORT.md without rerunning experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/b1_deployment'
MARKER = '<!-- measured-web-evidence -->'


def main():
    config = json.loads((ROOT / 'configs/straw_realtime.json').read_text(encoding='utf-8'))
    replay = json.loads((OUT / 'browser_replay.json').read_text(encoding='utf-8'))
    loads = [json.loads((OUT / f'http_{n}client.json').read_text(encoding='utf-8')) for n in (1, 2)]
    digest = config['checkpoint_sha256']
    if replay['checkpoint_sha256'] != digest or any(r['health']['checkpoint_sha256'] != digest for r in loads):
        raise ValueError('Web evidence belongs to a different checkpoint; rerun matching checks.')
    artifact = ROOT / config['checkpoint']
    lines = [MARKER, '## 7. Cấu hình thực sự phục vụ web', '',
        f"B1 đã fine-tune, đầu vào {config['size']} px, CUDA FP16 channels-last trên RTX 3050 Laptop. CPU phụ trách giải mã/chuẩn hóa và JSON. ONNX/PTQ/QAT CPU là các đối chứng; chúng không phải backend GPU của web.", '',
        f"Artifact FP16: `{config['checkpoint']}` ({artifact.stat().st_size / 2**20:.2f} MiB). SHA-256: `{digest}`. File trọng số không phải toàn bộ RAM/VRAM của dịch vụ.", '',
        '## 8. Tải HTTP thực tế trên localhost', '',
        'Mỗi bài đo khoảng 30 giây, nhịp mục tiêu 10 request/giây/client, gửi JPEG đã chuẩn bị. Không bao gồm camera, encode hay render trình duyệt; không chạy training đồng thời.', '',
        '| Client | Thành công / tổng | Thành công/giây | p50 HTTP (ms) | p95 (ms) | p99 (ms) | 429 / lỗi | Vượt 200 ms |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in loads:
        t = r['http_ms']
        lines.append(f"| {r['clients']} | {r['successful']}/{r['requests']} | {r['successful_per_second']:.2f} | {t['p50']:.2f} | {t['p95']:.2f} | {t['p99']:.2f} | {r['busy_429']}/{r['errors']} | {r['deadline_misses']} |")
    peak = max(r['server_peak_sampled_rss_mib'] for r in loads)
    lines += ['', f'Peak RSS lấy mẫu của tiến trình server: {peak:.1f} MiB. Bài đo ngắn này chưa chứng minh không rò bộ nhớ khi chạy dài.', '',
        'Server dùng một yêu cầu đang chạy và tối đa một yêu cầu chờ, ngân sách chờ 50 ms. Bản từ chối ngay khi GPU bận đã gây 297/594 request nhận 429 ở bài hai client; bản hàng đợi có hạn đạt 593/593 thành công ở bài đo sau. Đây là hai lượt đo chính sách, không phải bảo đảm mọi tải đều được nhận.', '',
        '## 9. Kiểm chứng qua canvas/JPEG của trình duyệt', '',
        f"Phát lại {replay['n']} ảnh validation qua ảnh nguồn proxy 640 px → canvas {config['size']} px → JPEG q90 → API thật → UI. Proxy được tạo độc lập nhãn từ ảnh gốc; đây là phát lại ảnh tĩnh, không phải camera hoặc dữ liệu mới.", '',
        f"Accuracy: {100*replay['accuracy']:.3f}%; Macro-F1: {100*replay['macro_f1']:.3f}%; FN={replay['fn']}, FP={replay['fp']}, sai loại lỗi={replay['wrong_defect_type']}. Cụ thể một Bent→Good và một Loose→Missing. Offline FP16 đạt Macro-F1 99,682%; đường trình duyệt thêm một lỗi phân loại loại lỗi. Không thay checkpoint/ngưỡng theo kết quả này.", '',
        '| Phạm vi đo | p50 (ms) | p95 (ms) | p99 (ms) |',
        '|---|---:|---:|---:|']
    for key, label in [('client_ms', 'Bắt đầu tải/giải mã nguồn → cập nhật UI'), ('http_ms', 'HTTP request/response'), ('encode_ms', 'Canvas resize/JPEG')]:
        t = replay['timings_ms'][key]
        lines.append(f"| {label} | {t['p50']:.2f} | {t['p95']:.2f} | {t['p99']:.2f} |")
    lines += ['', f"Có {replay['deadline_200ms_misses']}/{replay['n']} lượt vượt 200 ms: frame đầu 226,5 ms; chưa cô lập nguyên nhân. Không loại frame này khỏi thống kê. Không đo được độ trễ cảm biến/capture thật từ thí nghiệm phát lại.", '',
        'Một ví dụ tải ảnh gốc 3000×3000 trên trình duyệt trước đó mất khoảng 292,9 ms đến UI; không suy rộng kết quả proxy 640 px cho mọi file upload lớn.', '',
        '## 10. Quyết định theo hướng production', '',
        'Đích hiện tại là laptop NVIDIA, một hoặc hai luồng demo khoảng 10 Hz. Chọn GPU theo số đo toàn luồng và chất lượng trên phần cứng này; không ép mọi phép xử lý lên GPU. Giữ model trên GPU, nạp/warm-up một lần, batch=1 cho đường độ trễ thấp, kiểm tra SHA/config và phản hồi frame_id; không tạo hàng đợi vô hạn.', '',
        'Ma trận JPEG ở trên chưa gồm TensorRT hoặc ONNX Runtime CUDA. Nhánh RTSP/DeepStream đã thử TensorRT FP16 với H.264; xem artifacts/b1_stream/REPORT.md và docs/STRAW_RTSP_DEEPSTREAM.md. Phạm vi timing và tiền xử lý khác nên chưa phải đối chiếu GPU trên cùng tensor. Chưa thể tuyên bố runtime tối ưu tuyệt đối hoặc suy tốc độ INT8 GPU từ kết quả CPU. ONNX là định dạng; backend/provider xác định nơi thực thi.', '',
        'Bước nghiệm thu còn lại: camera thật và ánh sáng/khoảng cách thực; stress liên tục ít nhất 30 phút; theo dõi p95/p99, deadline-miss, nhiệt/clock, RAM/VRAM, lỗi và tải tăng. Nếu yêu cầu kiểm tra từng sản phẩm: cần product_id/trigger, chính sách quá tải rõ ràng và thử cơ cấu loại sản phẩm. frame_id chỉ đối chiếu request, chưa phải tracking sản phẩm. Không im lặng bỏ sản phẩm như cơ chế bỏ frame preview.', '',
        'Quy trình đo → tối ưu → đo lại theo [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html). Tài liệu mô tả cách làm; các số trong báo cáo lấy từ thí nghiệm cục bộ.', '']
    path = OUT / 'REPORT.md'
    base = path.read_text(encoding='utf-8').split(MARKER)[0].rstrip()
    path.write_text(base + '\n\n' + '\n'.join(lines), encoding='utf-8')
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
