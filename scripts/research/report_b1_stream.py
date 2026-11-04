"""Report engine/file quality separately from live RTSP timing."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.straw_global_local.runner import classification_metrics
from src.straw_global_local.data import file_hash
from src.straw_global_local.postprocess import predict_from_probabilities, counts
from sklearn.metrics import f1_score, confusion_matrix


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timing(rows):
    matched = np.array([r['decode_output_to_prediction_ms'] for r in rows], float)
    if not len(rows) or np.any(matched < 0) or not np.isfinite(matched).all():
        raise RuntimeError('Missing or invalid source-PTS timing; refusing latency claim')
    # Keep startup/first frame separately; report full sample and post-warmup.
    def quantile(a):
        return dict(n=len(a), **{f'p{q}':float(np.percentile(a,q)) for q in (50,95,99)}, max=float(a.max()))
    return dict(all_frames_ms=quantile(matched), after_first20_ms=quantile(matched[20:]) if len(rows)>20 else None)


def main():
    out = ROOT / 'artifacts' / os.environ.get('STRAW_STREAM_ARTIFACT','b1_stream')
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['onnx_sha256'] != file_hash(out / f"b1_rawrgb{manifest['size']}.onnx"):
        raise RuntimeError('ONNX changed since manifest')
    report = dict(source='validation-photo replay; no real camera/motion generalization claim',
                  checkpoint_sha256=manifest['checkpoint_sha256'], onnx_sha256=manifest['onnx_sha256'],
                  video_sha256=file_hash(out/'quality.mp4'), parity=manifest['parity'])
    if (out/'torch_h264_reference.json').exists():
        reference=json.loads((out/'torch_h264_reference.json').read_text())
        if reference['video_sha256']!=report['video_sha256'] or reference['checkpoint_sha256']!=manifest['checkpoint_sha256']:
            raise RuntimeError('PyTorch H264 reference belongs to different inputs')
        report['torch_h264_reference']=reference
    if (out/'b1_rtx3050_fp16.engine').exists():
        report['engine_sha256']=file_hash(out/'b1_rtx3050_fp16.engine')
        report['runtime']='NVIDIA TensorRT serialized engine, loaded by DeepStream nvinfer'
        report['runtime_engine']='b1_rtx3050_fp16.engine'
        report['onnx_role']='Build/export intermediate only; absent from nvinfer runtime config'
        report['engine_size_bytes']=(out/'b1_rtx3050_fp16.engine').stat().st_size
    if (out / 'file.jsonl').exists():
        rows = records(out / 'file.jsonl')
        summary = json.loads((out / 'file.jsonl.summary.json').read_text())
        if summary['failed'] or not summary['eos'] or summary['decoded']!=300 or summary['predicted']!=300:
            raise RuntimeError('File replay incomplete; cannot align validation IDs')
        if [r['frame_num'] for r in rows] != list(range(len(manifest['cases']))):
            raise RuntimeError('Frame order/count differs; refusing fabricated labels')
        probabilities = np.array([r['probabilities'] for r in rows])
        labels = np.array([r['label'] for r in manifest['cases']])
        raw_quality = classification_metrics(labels,np.log(np.maximum(probabilities,1e-30)))
        threshold = manifest.get('decision_threshold')
        predicted = predict_from_probabilities(probabilities, threshold)
        if any(r.get('class_index',int(predicted[i])) != int(predicted[i]) for i,r in enumerate(rows)):
            raise RuntimeError('DeepStream class decision differs from manifest policy')
        policy_quality = {**counts(labels,predicted),
                          'accuracy':float(np.mean(predicted==labels)),
                          'macro_f1':float(f1_score(labels,predicted,average='macro',labels=list(range(5)),zero_division=0)),
                          'confusion':confusion_matrix(labels,predicted,labels=list(range(5))).tolist()}
        policy_quality['wrong_defect_type'] = policy_quality['wrong_type']
        report['file'] = dict(summary=summary,raw_quality=raw_quality,quality=policy_quality,timing=timing(rows))
        quality = policy_quality
        anchor=json.loads((ROOT/'artifacts/b1_deployment/torch_fp32_exact/result.json').read_text())
        report['quality_gate']=dict(reference='B1 512 FP32, same 300 validation photos',
            reference_macro_f1=anchor['macro_f1'],max_drop=0.005,
            drop_percentage_points=100*(anchor['macro_f1']-quality['macro_f1']),
            passed=bool(quality['macro_f1']>=anchor['macro_f1']-0.005 and quality['fn']<=anchor['fn'] and quality['fp']<=anchor['fp']))
        errors=[dict(sample_id=c['sample_id'],true_class=manifest['classes'][int(labels[i])],
                     predicted_class=manifest['classes'][int(predicted[i])])
                for i,c in enumerate(manifest['cases']) if predicted[i]!=labels[i]]
        (out/'deepstream_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        if (out/'torch_h264_predictions.npz').exists():
            reference_predictions=np.load(out/'torch_h264_predictions.npz')
            if not np.array_equal(reference_predictions['labels'],labels):
                raise RuntimeError('Reference labels/order changed')
            report['file']['changed_vs_torch_h264']=int(np.sum(predicted!=reference_predictions['logits'].argmax(1)))
        np.savez_compressed(out/'deepstream_val_predictions.npz',probabilities=probabilities,labels=labels,
                            sample_ids=np.array([r['sample_id'] for r in manifest['cases']]))
        if out.name == 'b1_stream_512':
            legacy_path = ROOT/'artifacts/b1_stream/deepstream_val_predictions.npz'
            if legacy_path.exists():
                legacy = np.load(legacy_path)
                lookup = {str(sample):i for i,sample in enumerate(legacy['sample_ids'])}
                ids = [r['sample_id'] for r in manifest['cases']]
                if set(ids) != set(lookup): raise RuntimeError('Cannot pair old and new validation predictions')
                old_indices = [lookup[sample] for sample in ids]
                if not np.array_equal(legacy['labels'][old_indices],labels):
                    raise RuntimeError('Old and new labels differ')
                old_wrong = legacy['probabilities'][old_indices].argmax(1) != labels
                new_wrong = predicted != labels
                fixed = [ids[i] for i in np.where(old_wrong & ~new_wrong)[0]]
                regressed = [ids[i] for i in np.where(~old_wrong & new_wrong)[0]]
                changed = len(fixed)+len(regressed)
                report['paired_comparison'] = dict(old_errors=int(old_wrong.sum()),
                    new_errors=int(new_wrong.sum()), fixed_sample_ids=fixed,
                    regressed_sample_ids=regressed,
                    mcnemar_exact_two_sided_p=float(binomtest(len(fixed),changed,.5).pvalue) if changed else 1.0)
            report['deployment_decision'] = dict(promote=False,
                reason='Only validation-photo replay has been measured; no independent camera holdout or substantial paired-error reduction.')
    if (out / 'rtsp.jsonl').exists():
        rows = records(out/'rtsp.jsonl')
        summary = json.loads((out/'rtsp.jsonl.summary.json').read_text())
        if summary['failed']: raise RuntimeError('RTSP pipeline failed')
        report['rtsp'] = dict(summary=summary,timing=timing(rows),
            observed_fps=(len(rows)-1)/summary['observed_seconds'] if len(rows)>1 else None,
            note='No sample-label alignment claimed for RTSP; join/reconnect changes frame numbering. Timing starts after decode, excludes sensor/encoder/network/jitter/decode and browser.')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Nhánh RTSP / NVIDIA DeepStream', '',
        f"Checkpoint B1 seed 42, {manifest['size']} px; chuẩn hóa ImageNet được nhúng trong graph (RGB 0..255 NCHW). DeepStream nvinfer nạp engine TensorRT FP16, batch=1. Ngưỡng hậu xử lý: {manifest.get('decision_threshold')}. ONNX chỉ dùng để dựng engine.", '',
        f"Parity ONNX CPU: {manifest['parity']}", '',
        'Chất lượng dùng đúng 300 ảnh validation, mỗi ảnh một frame, H.264 640×640 CRF18, không B-frame; kiểm tra đủ frame và thứ tự trước khi gán nhãn. Video chỉ là phát lại ảnh tĩnh; không phải băng chuyền thật.', '']
    if 'torch_h264_reference' in report:
        q=report['torch_h264_reference']
        lines += [f"Đối chứng cùng clip H.264 bằng FFmpeg CPU decode/PIL resize/PyTorch CUDA FP16: Macro-F1 {q['macro_f1']*100:.3f}%; FN={q['fn']}, FP={q['fp']}, sai loại lỗi={q['wrong_defect_type']}. Chỉ đối chiếu chất lượng; không dùng latency của pipe WSL làm benchmark.", '']
    if 'file' in report:
        r=report['file']; q=r['quality']
        lines += [f"File DeepStream: Macro-F1 {q['macro_f1']*100:.3f}%; accuracy {q['accuracy']*100:.3f}%; FN={q['fn']}, FP={q['fp']}, sai loại lỗi={q['wrong_defect_type']}.", '']
        if 'changed_vs_torch_h264' in r:
            lines += [f"So với đối chứng PyTorch cùng clip, {r['changed_vs_torch_h264']}/300 nhãn thay đổi. Cả decoder và resize/backend đều khác; không quy chênh lệch chỉ cho FP16 hoặc TensorRT.", '']
    if 'rtsp' in report:
        r=report['rtsp']
        lines += [f"RTSP: {r['summary']['predicted']} frame; throughput quan sát {r['observed_fps']:.2f} FPS. Thống kê: `{r['summary']}`.", '',r['note'], '']
    lines += ['| Nguồn / phạm vi | Số frame | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |',
              '|---|---:|---:|---:|---:|---:|']
    for source in ('file','rtsp'):
        if source not in report: continue
        for key,label in [('all_frames_ms','tất cả'),('after_first20_ms','sau 20 frame đầu')]:
            t=report[source]['timing'][key]
            if t is None:continue
            lines.append(f"| {source}: sau decode → dự đoán, {label} | {t['n']} | {t['p50']:.2f} | {t['p95']:.2f} | {t['p99']:.2f} | {t['max']:.2f} |")
    lines += ['']
    for source in ('file','rtsp'):
        if source in report:
            lines += [f"{source}: thời gian từ khởi tạo ứng dụng đến dự đoán đầu tiên {report[source]['summary']['first_prediction_startup_ms']/1000:.2f} giây (tạo/nạp engine, kết nối nguồn và frame đầu).", '']
    if 'quality_gate' in report:
        gate=report['quality_gate']
        lines += [f"Cổng chất lượng kỹ thuật gốc: giảm Macro-F1 không quá 0,5 điểm phần trăm so với B1 512 FP32 ({gate['reference_macro_f1']*100:.3f}%), không tăng FN/FP. Nhánh DeepStream giảm {gate['drop_percentage_points']:.3f} điểm phần trăm; trạng thái: {'ĐẠT' if gate['passed'] else 'CHƯA ĐẠT'}. Đây là tiêu chí engineering trên validation, không phải kiểm định thống kê.", '',
                  'Cổng này dựa trên replay ảnh validation tĩnh; cần kiểm thêm camera và dữ liệu băng chuyền thật.', '']
    if 'deployment_decision' in report:
        lines += [f"Quyết định triển khai: KHÔNG THAY engine mặc định. {report['deployment_decision']['reason']}",
                  f"So sánh cặp với 384: {report.get('paired_comparison')}", '']
    cold=out/'cold_start/rtsp.jsonl.summary.json'
    if cold.exists() and 'rtsp' in report:
        previous=json.loads(cold.read_text())
        lines += [f"Lượt đầu trước sửa đường dẫn cache: startup RTSP {previous['first_prediction_startup_ms']/1000:.2f} giây vì engine bị dựng lại. Lượt kiểm tra sau sửa nạp đúng engine đã xác minh: {report['rtsp']['summary']['first_prediction_startup_ms']/1000:.2f} giây. Giữ dữ liệu lượt đầu trong cold_start/, không loại startup khỏi bằng chứng.", '']
    lines += ['Không so sánh trực tiếp độ trễ này với browser→UI hoặc HTTP round-trip. Timing từ pad trước nvstreammux đến output nvinfer bao gồm mux/preprocess/inference/copy output; chưa phải độ trễ camera→kết quả. Startup và first-frame được giữ riêng. FPS phát lại không chứng minh số sản phẩm kiểm tra/giây.', '',
              'Chưa kiểm tra camera IP thật, nhiễu/nhòe/ánh sáng, product_id/trigger, PLC hay chạy dài 30 phút. RTSP mô phỏng chỉ lắng nghe localhost trong WSL.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='file'},indent=2))
    if 'file' in report: print('QUALITY',report['file']['quality']['macro_f1'],report['file']['quality']['fn'],report['file']['quality']['fp'])


if __name__=='__main__': main()
