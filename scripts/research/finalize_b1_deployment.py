"""Final idle-machine latency recheck, recovery comparison and report generation.

Run only after both recovery jobs finish. Uses no test images or predictions.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from src.straw_global_local.data import file_hash
from src.straw_global_local.deployment import TorchRuntime,ORTRuntime,measure
from src.straw_global_local.deployment import decode
from src.straw_global_local.runner import classification_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/b1_deployment'


def main():
    torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/'artifacts/straw_global_local/data_v1/val.csv')
    ids=frame.groupby('label',sort=True).head(4).sample_id
    items_by_size={size:[(OUT/f'input_cache/{size}_q90'/f'{i}.jpg').read_bytes() for i in ids] for size in (384,512)}
    rows=[]
    checkpoint=ROOT/'artifacts/local_b1_check/B1_s42/best.pt'
    targets=[('gpu_fp32',512,lambda:TorchRuntime(checkpoint,precision='fp32')),
             ('gpu_fp16_cl',512,lambda:TorchRuntime(checkpoint,precision='fp16',channels_last=True)),
             ('onnx_cpu_fp32',512,lambda:ORTRuntime(OUT/'models/b1_512_fp32.onnx')),
             ('onnx_cpu_int8',512,lambda:ORTRuntime(OUT/'models/b1_512_int8.onnx'))]
    class ScriptRuntime:
        size=512
        def __init__(self): self.model=torch.jit.load(str(OUT/'recovery_qat_512_s42/best_int8.ts'),map_location='cpu').eval()
        @torch.inference_mode()
        def run(self,x): return self.model(torch.from_numpy(x)).numpy()
    if (OUT/'recovery_qat_512_s42/best_int8.ts').exists():targets.append(('qat_cpu_int8',512,ScriptRuntime))
    safe_folder=OUT/'recovery_finetune_384_s42_safety'
    recovered_folder=safe_folder if (safe_folder/'best_eligible.pt').exists() else OUT/'recovery_finetune_384_s42'
    recovered=recovered_folder/('best_eligible.pt' if (recovered_folder/'best_eligible.pt').exists() else 'best.pt')
    if recovered.exists():targets.append(('finetuned384_gpu_fp16_cl',384,lambda:TorchRuntime(recovered,size=384,precision='fp16',channels_last=True)))
    for name,size,factory in targets:
        runtime=factory();runtime.size=size
        if name.startswith(('gpu','finetuned')):torch.cuda.reset_peak_memory_stats()
        result={'name':name,'size':size,**measure(runtime,items_by_size[size],repeats=150,warmup=20)}
        if name.startswith(('gpu','finetuned')):
            result['peak_torch_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            result['weights_and_buffers_mib']=sum(t.numel()*t.element_size() for t in list(runtime.model.parameters())+list(runtime.model.buffers()))/2**20
        print(result,flush=True);rows.append(result)
        if name=='finetuned384_gpu_fp16_cl':
            logits=np.asarray([runtime.run(decode((OUT/'input_cache/384_q90'/f'{sample}.jpg').read_bytes(),384))[0] for sample in frame.sample_id])
            quality=classification_metrics(frame.label.to_numpy(),logits)
            (recovered_folder/'fp16_val.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
        del runtime
        if torch.cuda.is_available():torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT/'final_idle_benchmarks.csv',index=False)
    pilot=pd.read_csv(OUT/'summary.csv')
    reference=np.load(OUT/'torch_fp32_exact/val_predictions.npz')
    jpeg=np.load(OUT/'torch_fp32_jpeg512/val_predictions.npz')
    agreement=int(np.sum(reference['logits'].argmax(1)!=jpeg['logits'].argmax(1)))
    recoveries={}
    for name in ('recovery_finetune_384_s42','recovery_qat_512_s42','recovery_finetune_384_s42_safety'):
        path=OUT/name/'history.json'
        if path.exists():
            history=json.loads(path.read_text())
            best=max(history,key=lambda h:h['validation']['macro_f1'])
            recoveries[name]=dict(epochs_finished=len(history),best_epoch=best['epoch'],**best['validation'])
            eligible=[h for h in history if h.get('eligible')]
            if eligible:
                kept=max(eligible,key=lambda h:h['validation']['macro_f1'])
                recoveries[name+'_eligible']=dict(epochs_finished=len(history),best_epoch=kept['epoch'],**kept['validation'])
    calibration=json.loads((OUT/'temperature_crossfit.json').read_text())
    lines=['# Kết quả tối ưu B1 cho web', '',
        'So sánh runtime dùng seed 42, 300 ảnh validation (240 lỗi, 60 tốt); nhánh phục hồi 384 có kiểm chứng thêm seed 43/44. Không dùng test để chọn cấu hình mới.',
        'Đây là nghiên cứu mở rộng sau khi đã xem test cũ. Chưa có xác nhận độc lập trên camera/dữ liệu ngoài.', '',
        '## 1. Chất lượng và đầu vào', '',
        '| Cấu hình | Macro-F1 (%) | FN | FP | Sai loại lỗi | Ảnh gửi trung bình (KiB) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in pilot.itertuples():
        lines.append(f'| {r.name} | {r.macro_f1*100:.3f} | {r.fn} | {r.fp} | {r.wrong_defect_type} | {r.mean_payload_kib:.1f} |')
    lines+=['',f'JPEG 512 q90 mô phỏng PIL thay đổi {agreement}/300 nhãn argmax so với ảnh gốc. Canvas trình duyệt có encoder/nội suy khác, cần kiểm tra riêng.', '',
        '## 2. Kiểm tra tốc độ lại sau huấn luyện', '',
        'Batch=1, cùng 20 sample ID validation, JPEG 512 (hoặc 384 cho nhánh tương ứng), 20 warm-up và 150 lượt đo mỗi cấu hình. Không chạy training song song. Các số dưới đây chỉ là decode → logits → softmax trong tiến trình đo; chưa gồm browser/mạng/HTTP.', '',
        '| Runtime | Tiền xử lý p50 (ms) | Suy luận + chuyển dữ liệu p50 (ms) | Tổng p50 (ms) | Tổng p95 (ms) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['preprocess_p50_ms']:.2f} | {r['infer_with_transfer_p50_ms']:.2f} | {r['server_p50_ms']:.2f} | {r['server_p95_ms']:.2f} |")
    lines+=['', 'Các thời gian CPU trong summary.csv là pilot có tác vụ phục hồi chạy đồng thời; dùng bảng final_idle_benchmarks.csv để so sánh tốc độ cuối. Mô hình GPU không mặc nhiên nhanh hơn ở FP16.', '',
        '## 3. Phục hồi chất lượng', '',
        '| Phương án | Epoch đã chạy | Epoch tốt nhất (tính từ 0) | Macro-F1 (%) | FN | FP |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in recoveries.items():
        lines.append(f"| {name} | {r['epochs_finished']} | {r['best_epoch']} | {r['macro_f1']*100:.3f} | {r['fn']} | {r['fp']} |")
    lines+=['','Checkpoint phục hồi chọn theo Macro-F1. Khi chốt triển khai phải kiểm tra lại ràng buộc FN/FP; Macro-F1 cao hơn không tự động đủ điều kiện.', '',
        'QAT xuất TorchScript INT8 CPU dùng backend onednn ở môi trường này. So sánh PTQ ONNX với QAT TorchScript đồng thời thay đổi runtime; không thể quy toàn bộ chênh lệch latency cho QAT. Muốn kết luận riêng về QAT cần PTQ/QAT cùng runtime và backend.', '',
        '## 4. Hậu xử lý và web', '',
        f"Temperature scaling group cross-fit: NLL {calibration['before']['nll']:.4f} → {calibration['crossfit']['nll']:.4f}; Brier {calibration['before']['brier']:.5f} → {calibration['crossfit']['brier']:.5f}. Giữ T=1 nếu không cải thiện cả hai tiêu chí; không tự đổi ngưỡng.",
        'Web chạy B1 riêng ở localhost:8032, một luồng suy luận được warm-up, mỗi client một request đang xử lý, từ chối khi quá tải, trả frame_id và JSON nhỏ. Có đo capture→JSON và capture→cập nhật UI; chưa có dữ liệu video nhiều sản phẩm để bật smoothing/tracking.', '',
        '## 5. Kết luận và phần chưa xác nhận', '',
        'Tối ưu truyền/giải mã ảnh là nhánh triển vọng rõ ràng trên bộ này. Cấu hình nén phải qua kiểm tra chất lượng và đo trên đúng runtime, không chọn vì file nhỏ hoặc precision thấp.',
        'Nhánh fine-tune 384 đã kiểm tra seed 42/43/44; QAT mới có pilot seed 42. Chưa stress camera 30 phút; chưa xác nhận p95 của mọi client/mạng; chưa đánh giá test của cấu hình mới. Các kiểm tra HTTP ngắn và mẫu browser phải đọc riêng, không gọi là kiểm chứng dây chuyền thực tế.', '',
        'Chi tiết thiết kế, công thức, giới hạn và lệnh chạy: docs/STRAW_REALTIME_OPTIMIZATION.md.']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write={'jpeg_changed_predictions':agreement,'recoveries':recoveries,'checkpoint_sha256':file_hash(checkpoint)}
    (OUT/'final_results.json').write_text(json.dumps(write,indent=2),encoding='utf-8')
    seed_rows=[]
    for seed in (42,43,44):
        baseline=json.loads((ROOT/f'artifacts/local_b1_check/B1_s{seed}/complete.json').read_text())['validation']
        history=json.loads((OUT/f'recovery_finetune_384_s{seed}_safety/history.json').read_text())
        best=max(history,key=lambda h:h['validation']['macro_f1'])['validation']
        seed_rows.append(dict(seed=seed,baseline_macro_f1=baseline['macro_f1'],recovered_macro_f1=best['macro_f1'],baseline_fn=baseline['fn'],recovered_fn=best['fn'],baseline_fp=baseline['fp'],recovered_fp=best['fp']))
    pd.DataFrame(seed_rows).to_csv(OUT/'recovery_seed_comparison.csv',index=False)
    with (OUT/'REPORT.md').open('a',encoding='utf-8') as stream:
        stream.write('\n\n## 6. Kiểm chứng fine-tune 384 trên ba seed\n\n| Seed | B1 512 Macro-F1 (%) | Fine-tune 384 Macro-F1 (%) | FN trước/sau | FP trước/sau |\n|---|---:|---:|---:|---:|\n')
        for row in seed_rows:
            stream.write(f"| {row['seed']} | {100*row['baseline_macro_f1']:.3f} | {100*row['recovered_macro_f1']:.3f} | {row['baseline_fn']}/{row['recovered_fn']} | {row['baseline_fp']}/{row['recovered_fp']} |\n")
        stream.write('\nBa seed cùng tập validation, không phải ba bộ dữ liệu độc lập. Seed phục vụ web cố định 42; không chọn seed tốt nhất. Pipeline không ép deterministic CUDA; lượt tái lập seed 42 khác pilot trước, nguyên nhân cụ thể chưa được kiểm chứng riêng. Không tuyên bố lặp bitwise.\n')
    print('Saved '+str(OUT/'REPORT.md'),flush=True)


if __name__=='__main__':main()
