"""Resident TensorRT worker. Raw float32 RGB/CHW requests arrive only on ROI events.

Runs inside the existing NVIDIA container; uses TensorRT + libcudart directly,
without importing Torch, ONNX Runtime, NumPy or encoding an RTSP clip per image.
"""
import argparse
import ctypes as C
import ctypes.util
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import tensorrt as trt

from serving_contract import CLASSES, load_contract, check_identity


def atomic_json(path, data, optional=False):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    # DrvFS/Windows readers can momentarily hold the destination without
    # FILE_SHARE_DELETE. A telemetry read must never unload the resident GPU.
    for attempt in range(40):
        try:
            temporary.replace(path)
            return True
        except PermissionError:
            time.sleep(min(.005 * (attempt + 1), .05))
    temporary.unlink(missing_ok=True)
    if not optional:
        raise RuntimeError(f"Cannot publish {path.name}: Windows sharing lock persisted")
    return False


class Cuda:
    def __init__(self):
        self.lib = C.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
        signatures = {
            "cudaMalloc": ([C.POINTER(C.c_void_p), C.c_size_t], C.c_int),
            "cudaFree": ([C.c_void_p], C.c_int),
            "cudaHostAlloc": ([C.POINTER(C.c_void_p), C.c_size_t, C.c_uint], C.c_int),
            "cudaFreeHost": ([C.c_void_p], C.c_int),
            "cudaStreamCreate": ([C.POINTER(C.c_void_p)], C.c_int),
            "cudaStreamSynchronize": ([C.c_void_p], C.c_int),
            "cudaStreamDestroy": ([C.c_void_p], C.c_int),
            "cudaMemcpyAsync": ([C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed with CUDA code {code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--keepalive-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.keepalive_seconds < 0:
        raise ValueError("Keepalive interval must be nonnegative")
    identity = load_contract(args.engine.parent)
    size = identity["model_size"]
    threshold = identity["decision_threshold"]
    input_bytes = 3 * size * size * 4
    args.queue.mkdir(parents=True, exist_ok=True)
    lock = (args.queue / "worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Resident worker already owns the queue", flush=True)
        return
    status_path = args.queue / "runtime.json"
    generation = uuid.uuid4().hex
    status = dict(state="loading", generation=generation, pid=os.getpid(),
                  engine=args.engine.name, product_calls=0, warmup_calls=3, keepalive_calls=0,
                  heartbeat=time.time(), backend="TensorRT FP16 / CUDA", error="", **identity)
    atomic_json(status_path, status)
    cuda = None
    allocations = []
    stream = C.c_void_p()
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
        if engine is None:
            raise RuntimeError("Cannot deserialize the prepared TensorRT engine")
        context = engine.create_execution_context()
        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or outputs != ["probabilities"]:
            raise RuntimeError(f"Unexpected engine IO: {inputs}, {outputs}")
        if tuple(engine.get_tensor_shape(inputs[0])) != (1, 3, size, size):
            raise RuntimeError(f"Expected raw RGB input [1,3,{size},{size}]")
        if tuple(engine.get_tensor_shape(outputs[0])) != (1, len(CLASSES)):
            raise RuntimeError("Expected five output probabilities in manifest class order")
        if engine.get_tensor_dtype(inputs[0]) != trt.float32 or engine.get_tensor_dtype(outputs[0]) != trt.float32:
            raise RuntimeError("Expected float32 engine IO")
        cuda = Cuda()
        cuda.call("cudaStreamCreate", C.byref(stream))
        device_in, device_out, host_in, host_out = [C.c_void_p() for _ in range(4)]
        for pointer, bytes_needed, host in ((device_in, input_bytes, False), (device_out, 20, False),
                                            (host_in, input_bytes, True), (host_out, 20, True)):
            if host:
                cuda.call("cudaHostAlloc", C.byref(pointer), bytes_needed, 0)
            else:
                cuda.call("cudaMalloc", C.byref(pointer), bytes_needed)
            allocations.append((pointer, host))
        context.set_tensor_address(inputs[0], device_in.value)
        context.set_tensor_address(outputs[0], device_out.value)

        last_infer = time.monotonic()
        def infer():
            nonlocal last_infer
            started = time.perf_counter()
            cuda.call("cudaMemcpyAsync", device_in, host_in, input_bytes, 1, stream)
            if not context.execute_async_v3(stream_handle=stream.value):
                raise RuntimeError("TensorRT enqueue failed")
            cuda.call("cudaMemcpyAsync", host_out, device_out, 20, 2, stream)
            cuda.call("cudaStreamSynchronize", stream)
            values = list((C.c_float * 5).from_address(host_out.value))
            if not all(0 <= value <= 1.001 for value in values) or abs(sum(values) - 1) > .015:
                raise RuntimeError(f"Invalid probabilities: {values}")
            last_infer = time.monotonic()
            return values, (time.perf_counter() - started) * 1000

        C.memset(host_in, 0, input_bytes)
        for _ in range(3):
            infer()
        status["state"] = "ready"
        status["loaded_at"] = time.time()
        print(f"TENSORRT RESIDENT READY generation={generation}", flush=True)
        last_heartbeat = 0
        while True:
            if time.time() - last_heartbeat > .5:
                status["heartbeat"] = time.time()
                atomic_json(status_path, status, optional=True)
                last_heartbeat = time.time()
            requests = sorted(args.queue.glob("*.f32"), key=lambda p: p.stat().st_mtime_ns)
            for request in requests:
                result_path = request.with_suffix(".json")
                if result_path.exists():
                    request.unlink(missing_ok=True)
                    continue
                try:
                    data = request.read_bytes()
                    event = json.loads(request.with_suffix(".event.json").read_text(encoding="utf-8"))
                    check_identity(event, identity)
                    if event.get("generation") != generation:
                        raise ValueError("Request belongs to a previous worker generation; resubmit the image")
                    if len(data) != input_bytes:
                        raise ValueError("Invalid raw RGB tensor byte count")
                    input_sha = hashlib.sha256(data).hexdigest()
                    if input_sha != event.get("input_sha256"):
                        raise ValueError("Raw tensor hash differs from submitted image")
                    C.memmove(host_in, data, input_bytes)
                    values, elapsed = infer()
                    status["product_calls"] += 1
                    best = max(range(5), key=values.__getitem__)
                    if threshold is not None:
                        best = max(range(1,5), key=values.__getitem__) if 1-values[0] >= threshold else 0
                    atomic_json(result_path, dict(status="completed", request_id=request.stem,
                        generation=generation, probabilities=dict(zip(CLASSES, values)),
                        **{"class": CLASSES[best]}, defective=best != 0, confidence=values[best],
                        tensor_transfer_inference_ms=elapsed, completed_at=time.time(),
                        input_sha256=input_sha, **identity))
                    print(f"ROI INFER {request.stem} {CLASSES[best]} {elapsed:.3f}ms", flush=True)
                except Exception as exc:
                    atomic_json(result_path, dict(status="error", error=str(exc), request_id=request.stem))
                finally:
                    request.unlink(missing_ok=True)
            if args.keepalive_seconds and not requests and time.monotonic() - last_infer >= args.keepalive_seconds:
                # Keep the resident CUDA context and GPU clocks active between
                # sparse conveyor events; this does not publish a prediction.
                infer()
                status["keepalive_calls"] += 1
            time.sleep(.02)
    except Exception as exc:
        status.update(state="error", error=str(exc), heartbeat=time.time())
        atomic_json(status_path, status)
        raise
    finally:
        if cuda:
            for pointer, host in reversed(allocations):
                cuda.call("cudaFreeHost" if host else "cudaFree", pointer)
            if stream.value:
                cuda.call("cudaStreamDestroy", stream)


if __name__ == "__main__":
    main()
