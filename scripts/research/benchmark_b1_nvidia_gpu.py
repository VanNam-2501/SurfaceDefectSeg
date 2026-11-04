"""GPU-only B1 deployment comparison on the RTX NVIDIA target.

Uses the selected web checkpoint and validation/train split only:
PyTorch CUDA FP32/FP16, ONNX Runtime CUDA FP32/FP16, TensorRT INT8 QDQ.
"""
from __future__ import annotations
import argparse, gc, json, os, platform, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/b1_nvidia_gpu"
DEP = ROOT / "artifacts/b1_deployment"
CHECKPOINT = DEP / "models/b1_web384_fp16.pt"
SIZE, QUALITY = 384, 90
ORT_GPU = DEP / "onnxruntime_gpu_1_26"
ORT_CONVERTER = DEP / "onnxconverter_common"
TRT_ROOT = DEP / "tensorrt_cuda12_11_4"

sys.path.insert(0, str(ROOT))
from src.straw_global_local.data import file_hash
from src.straw_global_local.deployment import TensorB1, TorchRuntime, decode
from src.straw_global_local.runner import classification_metrics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def source(row):
    return ROOT / "data/straw_defects_raw" / row.relative_path


def payload_384(sample_id, fallback_row=None):
    # Create the target-size proxy directly from the original image, then cache
    # these exact bytes so resumed candidates use identical input tensors.
    cache = OUT / "input_cache/384_q90" / f"{sample_id}.jpg"
    if cache.exists():
        return cache.read_bytes()
    if fallback_row is None:
        raise FileNotFoundError(f"No cached deployment input for {sample_id}")
    from src.straw_global_local.deployment import packet
    payload = packet(source(fallback_row), SIZE, QUALITY)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return payload


def load_ort():
    # Import torch first so its CUDA 12.8/cuDNN 9 DLLs are available to ORT.
    sys.path.insert(0, str(ORT_GPU))
    import onnxruntime as ort
    ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"ORT CUDA EP unavailable: {ort.get_available_providers()}")
    return ort


class OrtCudaRuntime:
    def __init__(self, path, precision):
        self.ort = load_ort()
        self.size = SIZE
        opts = self.ort.SessionOptions()
        opts.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_profiling = True
        provider_options = {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC",
                            "use_tf32": "1"}
        self.session = self.ort.InferenceSession(
            str(path), sess_options=opts,
            providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"])
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA EP not active: {self.session.get_providers()}")
        self.input = self.session.get_inputs()[0].name
        self.profile = None
        self.precision = precision

    def run(self, x):
        return self.session.run(None, {self.input: x})[0]

    def finish_profile(self):
        profile_path = self.session.end_profiling()
        events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        counts = {}
        for event in events:
            provider = event.get("args", {}).get("provider")
            if provider:
                counts[provider] = counts.get(provider, 0) + 1
        self.profile = {"profile_file": str(Path(profile_path).resolve()),
                        "node_events_by_provider": counts}
        return self.profile


class TensorRtRuntime:
    def __init__(self, engine_path):
        self.size = SIZE
        self.trt, self.cuda = load_trt()
        logger = self.trt.Logger(self.trt.Logger.ERROR)
        runtime = self.trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT failed to deserialize INT8 engine")
        self.context = self.engine.create_execution_context()
        self.runtime = runtime
        self.input_name = next(self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == self.trt.TensorIOMode.INPUT)
        self.output_name = next(self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                                if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == self.trt.TensorIOMode.OUTPUT)
        self.context.set_input_shape(self.input_name, (1, 3, SIZE, SIZE))
        self.stream = torch.cuda.Stream()
        self.output = torch.empty(tuple(self.engine.get_tensor_shape(self.output_name)),
                                  dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.output_name, int(self.output.data_ptr()))

    def run(self, x):
        with torch.cuda.stream(self.stream):
            input_tensor = torch.from_numpy(np.ascontiguousarray(x)).to(device="cuda", dtype=torch.float32)
            self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT enqueue failed")
        self.stream.synchronize()
        return self.output.float().cpu().numpy()


def load_trt():
    # TensorRT pip runtime ships its DLLs in sibling package directories.
    candidates = [TRT_ROOT, TRT_ROOT / "tensorrt", TRT_ROOT / "tensorrt_bindings"]
    candidates += list(TRT_ROOT.glob("tensorrt_libs*"))
    for directory in candidates:
        if directory.is_dir():
            try:
                os.add_dll_directory(str(directory.resolve()))
            except OSError:
                pass
    sys.path.insert(0, str(TRT_ROOT))
    import tensorrt as trt
    return trt, None


def export_onnx(val_payload):
    import onnx
    sys.path.insert(0, str(ORT_CONVERTER))
    from onnxconverter_common import float16
    model = TensorB1(CHECKPOINT).eval().float()
    x = decode(val_payload, SIZE)
    fp32_path = OUT / "models/b1_web384_fp32.onnx"
    fp16_path = OUT / "models/b1_web384_fp16.onnx"
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    if not fp32_path.exists():
        torch.onnx.export(model, torch.from_numpy(x), str(fp32_path), opset_version=17,
                          input_names=["image"], output_names=["logits"], dynamo=False)
    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)
    if not fp16_path.exists():
        half_graph = float16.convert_float_to_float16(graph, keep_io_types=True)
        onnx.checker.check_model(half_graph)
        onnx.save(half_graph, str(fp16_path))
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).numpy()
    ort = load_ort()
    sess = ort.InferenceSession(str(fp32_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    actual = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    parity = float(np.max(np.abs(expected - actual)))
    changed = int(expected.argmax(axis=1)[0] != actual.argmax(axis=1)[0])
    if changed:
        raise RuntimeError(f"ONNX CUDA export changed top-1 on the parity sample: {parity}")
    write_json(OUT / "models/export_parity.json", {
        "checkpoint_sha256": file_hash(CHECKPOINT), "input_size": SIZE,
        "ort_cuda_fp32_max_abs_logit_error": parity,
        "ort_cuda_fp32_changed_top1_on_parity_sample": changed,
        "fp16_model_bytes": fp16_path.stat().st_size,
        "fp32_model_bytes": fp32_path.stat().st_size})
    return fp32_path, fp16_path


def quantize_qdq(fp32_path, train, payload_by_id):
    # Static QDQ calibration uses only 200 balanced Train samples (40/class).
    import onnx
    ort = load_ort()
    from onnxruntime.quantization import (CalibrationDataReader, quantize_static,
        QuantType, QuantFormat, CalibrationMethod)
    out = OUT / "models/b1_web384_int8_conv_symmetric_qdq.onnx"
    calibration = train.groupby("label", sort=True).head(40)
    write_json(OUT / "models/int8_calibration_manifest.json", {
        "split": "train", "n": len(calibration), "per_class": 40,
        "sample_ids": calibration.sample_id.tolist(), "quality": QUALITY,
        "quantized_ops": ["Conv"], "classifier": "FP32", "symmetric_int8": True,
        "checkpoint_sha256": file_hash(CHECKPOINT)})
    if not out.exists():
        temp_root = OUT / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temp_root)
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.rows = iter(calibration.itertuples())
            def get_next(self):
                row = next(self.rows, None)
                if row is None:
                    return None
                payload = payload_by_id[row.sample_id]
                return {"image": decode(payload, SIZE)}
        quantize_static(str(fp32_path), str(out), Reader(),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                        op_types_to_quantize=["Conv"],
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    onnx.checker.check_model(onnx.load(str(out)))
    return out


def build_trt_int8(qdq_path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(qdq_path))
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    kept_nodes = []
    restored_biases = 0
    for node in model.graph.node:
        if (node.op_type == "DequantizeLinear" and node.input[0] in initializers
                and initializers[node.input[0]].dtype == np.int32):
            quantized = initializers[node.input[0]].astype(np.float32)
            scale = initializers[node.input[1]].astype(np.float32)
            zero = (initializers[node.input[2]].astype(np.float32)
                    if len(node.input) > 2 and node.input[2] in initializers else 0.0)
            restored = (quantized - zero) * scale
            model.graph.initializer.append(
                numpy_helper.from_array(restored.astype(np.float32), name=node.output[0]))
            restored_biases += 1
        else:
            kept_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    used = {name for node in model.graph.node for name in node.input}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    compatible = OUT / "models/b1_web384_int8_conv_trt_qdq.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, str(compatible))
    write_json(OUT / "models/trt_qdq_compatibility.json", {
        "original_qdq": str(qdq_path.resolve()),
        "trt_graph": str(compatible.resolve()),
        "int32_bias_qdq_restored_to_float": restored_biases,
        "reason": "TensorRT importer requires Q/DQ data types supported for quantized tensors; bias dequantization is numerically restored to FP32."})
    qdq_path = compatible
    trt, _ = load_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 10/11 always uses explicit dimensions; the former
    # EXPLICIT_BATCH network flag was removed from the Python API.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    graph = qdq_path.read_bytes()
    if not parser.parse(graph):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX QDQ parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if hasattr(trt.BuilderFlag, "INT8"):
        config.set_flag(trt.BuilderFlag.INT8)
    config.builder_optimization_level = 0
    # Q/DQ nodes carry explicit calibrated quantization scales.
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT INT8 engine build returned None")
    path = OUT / "engines/b1_web384_int8.plan"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(engine))
    write_json(OUT / "engines/tensorrt_build.json", {
        "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0),
        "source_qdq": str(qdq_path.resolve()), "source_sha256": file_hash(qdq_path),
        "engine_sha256": file_hash(path), "engine_bytes": path.stat().st_size,
        "input": [1, 3, SIZE, SIZE], "precision": "INT8 Conv + FP32 classifier via explicit Q/DQ",
        "workspace_bytes": 1 << 30})
    return path


def build_trt_fp16(fp16_path):
    trt, _ = load_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(fp16_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX FP32 parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT FP16 engine build returned None")
    path = OUT / "engines/b1_web384_fp16.plan"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(engine))
    write_json(OUT / "engines/tensorrt_fp16_build.json", {
        "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0),
        "source_onnx": str(fp16_path.resolve()), "source_sha256": file_hash(fp16_path),
        "engine_sha256": file_hash(path), "engine_bytes": path.stat().st_size,
        "input": [1, 3, SIZE, SIZE], "precision": "FP16 ONNX nodes, FP32 I/O"})
    return path


def evaluate_candidate(name, runtime, val, payloads, metadata, repeats=300):
    logits = []
    for i, payload in enumerate(payloads):
        logits.append(runtime.run(decode(payload, SIZE))[0])
        if (i + 1) % 100 == 0:
            print(f"{name}: validation {i+1}/{len(val)}", flush=True)
    logits = np.asarray(logits)
    metrics = classification_metrics(val.label.to_numpy(), logits)
    metrics.update(metadata)
    # Benchmark same 20 balanced sample IDs after 30 warm-ups.
    indices = val.reset_index().groupby("label", sort=True).head(4)["index"].tolist()
    times = []
    for i in range(30 + repeats):
        payload = payloads[indices[i % len(indices)]]
        t0 = time.perf_counter()
        x = decode(payload, SIZE)
        t1 = time.perf_counter()
        result = runtime.run(x)
        t2 = time.perf_counter()
        probabilities = np.exp(result.astype(np.float64) - result.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        int(probabilities.argmax(axis=-1)[0])
        t3 = time.perf_counter()
        if i >= 30:
            times.append([t1-t0, t2-t1, t3-t2, t3-t0])
    values = np.asarray(times) * 1000
    for i, part in enumerate(["preprocess", "infer_and_transfer", "postprocess", "end_to_end"]):
        metrics[f"{part}_p50_ms"] = float(np.quantile(values[:, i], .50))
        metrics[f"{part}_p95_ms"] = float(np.quantile(values[:, i], .95))
        metrics[f"{part}_p99_ms"] = float(np.quantile(values[:, i], .99))
    metrics.update(name=name, n=len(val), input_size=SIZE, warmup=30, benchmark_repeats=repeats)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "val_predictions.npz", logits=logits,
                        labels=val.label.to_numpy(), sample_ids=val.sample_id.to_numpy(dtype=str))
    if hasattr(runtime, "finish_profile"):
        metrics.update(runtime.finish_profile())
    write_json(folder / "result.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--skip-tensorrt", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--trt-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This study requires an NVIDIA CUDA GPU")
    print("Starting GPU deployment study", flush=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    val = pd.read_csv(ROOT / "artifacts/straw_global_local/data_v1/val.csv")
    train = pd.read_csv(ROOT / "artifacts/straw_global_local/data_v1/train.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = train.groupby("label", sort=True).head(40)
    print(f"Loaded manifests: val={len(val)}, int8 calibration={len(calibration)}", flush=True)
    for row in val.itertuples():
        if file_hash(source(row)) != row.sha256:
            raise RuntimeError(f"Validation source hash mismatch: {row.sample_id}")
        if (row.Index + 1) % 50 == 0:
            print(f"Verified validation hashes: {row.Index+1}/{len(val)}", flush=True)
    print("Validation hashes verified; encoding JPEG inputs", flush=True)
    val_payloads = []
    for i, row in enumerate(val.itertuples(), 1):
        val_payloads.append(payload_384(row.sample_id, row))
        if i % 50 == 0:
            print(f"Prepared validation inputs: {i}/{len(val)}", flush=True)
    train_by_id = {}
    for i, row in enumerate(calibration.itertuples(), 1):
        if file_hash(source(row)) != row.sha256:
            raise RuntimeError(f"Calibration source hash mismatch: {row.sample_id}")
        train_by_id[row.sample_id] = payload_384(row.sample_id, row)
        if i % 50 == 0:
            print(f"Prepared calibration images: {i}/{len(calibration)}", flush=True)
    print("Inputs ready; beginning PyTorch CUDA comparisons", flush=True)
    write_json(OUT / "study.json", {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(), "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": file_hash(CHECKPOINT),
        "val_manifest_sha256": file_hash(ROOT / "artifacts/straw_global_local/data_v1/val.csv"),
        "validation_only": True, "test_used": False,
        "input": "original -> EXIF transpose -> RGB -> bilinear 384 -> JPEG q90 -> ImageNet normalize; same bytes for every runtime",
        "benchmark": f"batch 1, 30 warmups, {args.repeats} measured repeats, decode to logits/softmax on localhost GPU"})
    if args.prepare_only:
        print("Prepared and cached GPU benchmark inputs:", OUT / "input_cache/384_q90", flush=True)
        return
    results = []
    # Compare GPU runtimes on exactly the selected web checkpoint and JPEG384 inputs.
    if args.trt_only:
        for candidate in ("pytorch_cuda_fp32", "pytorch_cuda_fp16",
                          "pytorch_cuda_fp16_channels_last", "onnxruntime_cuda_fp32",
                          "onnxruntime_cuda_fp16"):
            result_path = OUT / candidate / "result.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        for name, precision, channels in [
            ("pytorch_cuda_fp32", "fp32", False),
            ("pytorch_cuda_fp16", "fp16", False),
            ("pytorch_cuda_fp16_channels_last", "fp16", True)]:
            runtime = TorchRuntime(CHECKPOINT, SIZE, precision, channels_last=channels, autotune=True)
            results.append(evaluate_candidate(name, runtime, val, val_payloads,
                 {"runtime": "PyTorch CUDA", "precision": precision, "channels_last": channels}, args.repeats))
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
    fp32 = OUT / "models/b1_web384_fp32.onnx"
    fp16 = OUT / "models/b1_web384_fp16.onnx"
    if not args.trt_only:
        fp32, fp16 = export_onnx(val_payloads[0])
        for name, model, precision in [
            ("onnxruntime_cuda_fp32", fp32, "fp32"),
            ("onnxruntime_cuda_fp16", fp16, "fp16 internal, float32 I/O")]:
            runtime = OrtCudaRuntime(model, precision)
            results.append(evaluate_candidate(name, runtime, val, val_payloads,
                 {"runtime": "ONNX Runtime CUDA EP", "precision": precision,
                  "providers": runtime.session.get_providers()}, args.repeats))
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
    if not args.skip_tensorrt:
        trt_fp16 = build_trt_fp16(fp16)
        runtime = TensorRtRuntime(trt_fp16)
        results.append(evaluate_candidate("tensorrt_cuda_fp16", runtime, val, val_payloads,
             {"runtime": "TensorRT CUDA", "precision": "FP16 ONNX nodes, FP32 I/O",
              "engine_bytes": trt_fp16.stat().st_size}, args.repeats))
        try:
            qdq = quantize_qdq(fp32, train, train_by_id)
            engine_path = build_trt_int8(qdq)
            runtime = TensorRtRuntime(engine_path)
            results.append(evaluate_candidate("tensorrt_cuda_int8_qdq", runtime, val, val_payloads,
                 {"runtime": "TensorRT CUDA", "precision": "INT8 Conv + FP32 classifier QDQ",
                  "engine_bytes": engine_path.stat().st_size}, args.repeats))
        except Exception as exc:
            write_json(OUT / "engines/tensorrt_int8_build_failure.json", {
                "exception": repr(exc), "stage": "QDQ parse/build/runtime",
                "note": "A failed TensorRT build is not reported as an INT8 inference result."})
            print("TensorRT INT8 build unavailable; recorded failure and kept successful GPU candidates.", flush=True)
    pd.DataFrame([{k:v for k,v in row.items() if np.isscalar(v) or v is None}
                  for row in results]).to_csv(OUT / "summary.csv", index=False)
    print("Completed GPU study:", OUT, flush=True)


if __name__ == "__main__":
    main()
