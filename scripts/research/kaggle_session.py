"""Budgeted sequential Kaggle training, verified portable snapshots, no cloud credentials.

The model/training code is unchanged. Each call trains one complete epoch using
the existing exact-resume mechanism. This lets the session stop between epochs
without changing the declared 40-epoch cosine schedule or patience counter.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

RESUME_NAME = "straw_resume.zip"
STATE_DIR = Path("artifacts/straw_global_local")
PROTOCOL_REL = STATE_DIR / "data_v1"
CACHE_REL = STATE_DIR / "cache_v1"
OUTPUT_REL = STATE_DIR / "confirmatory_kaggle_v1"
IDENTITY_REL = Path("kaggle_session_identity.json")
SESSION_REL = Path("kaggle_session_status.json")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def safe_name(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def checked_manifest(value, kind):
    if value.get("kind") != kind or value.get("schema") != 1:
        raise ValueError(f"Expected {kind} schema 1")
    plain = {k: v for k, v in value.items() if k != "fingerprint"}
    if canonical_hash(plain) != value.get("fingerprint"):
        raise ValueError("Manifest fingerprint mismatch")
    for name, record in value["files"].items():
        safe_name(name)
        if not isinstance(record["bytes"], int) or record["bytes"] < 0 or len(record["sha256"]) != 64:
            raise ValueError("Invalid file record")
    return value


def copy_verified(stream, target, record):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".copying")
    value, size = hashlib.sha256(), 0
    try:
        with temp.open("wb") as destination:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                destination.write(chunk)
                value.update(chunk)
                size += len(chunk)
        if size != record["bytes"] or value.hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {target.name}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def materialize_bundle(source, project):
    """Stable absolute image paths are necessary for frozen CSV fingerprints."""
    source, project = Path(source), Path(project)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        manifest = checked_manifest(
            json.loads(archive.read("bundle_manifest.json")) if archive else read_json(source / "bundle_manifest.json"),
            "straw-kaggle-all-in-one")
        if archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest["files"]) | {"bundle_manifest.json"}:
                raise ValueError("Unexpected/duplicate files in bundle")
        for index, (name, record) in enumerate(manifest["files"].items(), 1):
            target = project / safe_name(name)
            if target.is_symlink():
                raise ValueError("Symlink target rejected")
            if target.exists() and target.stat().st_size == record["bytes"] and sha256(target) == record["sha256"]:
                continue
            with archive.open(name) if archive else (source / name).open("rb") as stream:
                copy_verified(stream, target, record)
            if index % 200 == 0:
                print(f"Verified/copied bundle files {index}/{len(manifest['files'])}", flush=True)
        return manifest
    finally:
        if archive:
            archive.close()


def snapshot_files(project):
    project = Path(project)
    paths = [project / IDENTITY_REL, project / SESSION_REL, project / "kaggle_session.log"]
    for directory in (PROTOCOL_REL, CACHE_REL, OUTPUT_REL):
        paths += list((project / directory).rglob("*"))
    return sorted({p for p in paths if p.is_file() and not p.name.endswith((".tmp", ".copying"))})


def make_snapshot(project, destination, bundle_fingerprint):
    """ZIP_STORED avoids recompressing model weights and already compressed NPZs."""
    project, destination = Path(project), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = snapshot_files(project)
    # A previous snapshot is replaceable because authoritative live checkpoints
    # remain intact. Avoid keeping two multi-GB archives in a 20-GB output area.
    required = sum(p.stat().st_size for p in paths) + 128 * 1024**2
    if shutil.disk_usage(destination.parent).free < required and destination.exists():
        destination.unlink()
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError("Not enough disk for resume ZIP; preserve the live project output directory")
    temp = destination.with_suffix(".writing.zip")
    records = {}
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in paths:
                if path.is_symlink():
                    raise ValueError("Symlink snapshot member rejected")
                name = path.relative_to(project).as_posix()
                value, size = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        value.update(chunk)
                        size += len(chunk)
                records[name] = {"bytes": size, "sha256": value.hexdigest()}
            manifest = {"schema": 1, "kind": "straw-kaggle-resume", "files": records,
                        "bundle": bundle_fingerprint, "project_root": str(project.resolve()),
                        "note": "Own trusted checkpoints only; raw images and pretrained weights come from the input bundle."}
            manifest["fingerprint"] = canonical_hash(manifest)
            archive.writestr("resume_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        temp.replace(destination)
        print(f"RESUME ZIP: {destination} ({destination.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        return manifest
    finally:
        temp.unlink(missing_ok=True)


def restore_snapshot(source, project, bundle_fingerprint):
    """Verify everything in a staging directory before loading any torch pickle."""
    project = Path(project)
    source = Path(source)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        manifest = checked_manifest(json.loads(archive.read("resume_manifest.json")) if archive else
                                    read_json(source / "resume_manifest.json"), "straw-kaggle-resume")
        if manifest["bundle"] != bundle_fingerprint or manifest["project_root"] != str(project.resolve()):
            raise ValueError("Resume bundle/root mismatch; keep the same bundle and PROJECT_ROOT")
        if archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest["files"]) | {"resume_manifest.json"}:
                raise ValueError("Unexpected/duplicate files in resume archive")
        allowed_roots = (PROTOCOL_REL, CACHE_REL, OUTPUT_REL)
        for name in manifest["files"]:
            relative = Path(safe_name(name))
            if relative not in (IDENTITY_REL, SESSION_REL, Path("kaggle_session.log")) and not any(relative.is_relative_to(root) for root in allowed_roots):
                raise ValueError("Unexpected resume path")
            if (project / relative).exists():
                raise FileExistsError(f"Refusing to overwrite live progress: {relative}")
        with tempfile.TemporaryDirectory(prefix="straw-resume-", dir=project.parent) as staging:
            staged = Path(staging)
            for name, record in manifest["files"].items():
                with archive.open(name) if archive else (source / name).open("rb") as stream:
                    copy_verified(stream, staged / name, record)
            for name in manifest["files"]:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged / name), str(target))
        return manifest
    finally:
        if archive:
            archive.close()


def find_resume(input_root, requested):
    if requested == "":
        return None
    if requested is not None:
        path = Path(requested)
        if not path.is_file() and not (path / "resume_manifest.json").is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path(input_root).rglob(RESUME_NAME))
    if not candidates:
        candidates = sorted({p.parent for p in Path(input_root).rglob("resume_manifest.json")})
    if len(candidates) > 1:
        raise ValueError(f"Multiple resume archives: set RESUME_FROM explicitly: {candidates}")
    return candidates[0] if candidates else None


def discover_dataset(project, input_root, requested=None):
    """Find the image tree even if Kaggle changes its mount nesting/slug."""
    with (Path(project) / "artifacts/straw_thesis/data_v2/iid/inventory.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    relatives = [PurePosixPath(row["relative_path"]) for row in rows]
    prefixes = {p.parts[0] for p in relatives}
    if len(prefixes) != 1:
        raise ValueError("Expected a single original dataset tree")
    prefix = next(iter(prefixes))
    tails = [Path(*p.parts[1:]) for p in relatives]
    if requested:
        root = Path(requested)
        candidates = [root, root / prefix]
    else:
        root = Path(input_root)
        candidates = list(root.rglob(prefix))
        # Also supports a dataset whose upload omitted the outermost folder.
        candidates += [p.parent for p in root.rglob(tails[0].parts[0]) if p.is_dir()]
    valid = sorted({p.resolve() for p in candidates if p.is_dir() and all((p / t).is_file() for t in tails)})
    if len(valid) != 1:
        raise ValueError(f"Found {len(valid)} matching image datasets ({valid}). Set RAW_ROOT to the original image folder; expected {len(rows)} paths such as {tails[0]}.")
    return valid[0], prefix


def bind_dataset(project, image_tree, prefix):
    """Linux symlink reads input in place while CSV paths stay fixed across sessions."""
    directory = Path(project) / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / prefix
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Expected a generated dataset link, found an existing directory: {link}")
    os.symlink(str(image_tree), str(link), target_is_directory=True)
    return directory, link


def prepare_stable_protocol(project, protocol, image_root):
    """prepare() resolves symlinks; freeze stable input aliases before training."""
    from src.straw_global_local import data
    with tempfile.TemporaryDirectory(prefix="straw-protocol-", dir=Path(project)) as temp:
        temporary = Path(temp) / "protocol"
        data.prepare(Path(project) / "artifacts/straw_thesis/data_v2/iid", temporary, image_root)
        metadata, frames = data.load_protocol(temporary)
        for frame in frames.values():
            frame["image_path"] = [str(Path(image_root) / safe_name(relative)) for relative in frame.relative_path]
        return data.save_protocol(frames, protocol, metadata["context"])


def ensure_pretrained(input_root, requested=None):
    """Keep Kaggle's torch pair; use official cached/input weights or download once."""
    import torch
    from torchvision.models import ResNet18_Weights
    target = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if requested:
        candidates = [Path(requested)]
    elif target.is_file():
        candidates = [target]
    else:
        candidates = sorted(Path(input_root).rglob(target.name))
    if candidates:
        if len(candidates) > 1:
            raise ValueError("Multiple pretrained files; set PRETRAINED_FILE explicitly")
        source = candidates[0]
        if not sha256(source).startswith("f37072fd"):
            raise ValueError("Expected official resnet18-f37072fd.pth")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    else:
        try:
            ResNet18_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        except Exception as exc:
            raise RuntimeError("Enable Kaggle Internet to download official ResNet18 weights once, or attach resnet18-f37072fd.pth and set PRETRAINED_FILE. No random-init fallback.") from exc


def verify_originals(protocol):
    from src.straw_global_local import data
    _, frames = data.load_protocol(protocol)
    count = 0
    for frame in frames.values():
        for row in frame.itertuples():
            if sha256(row.image_path) != row.sha256:
                raise ValueError(f"Mounted original differs from frozen dataset: {row.image_path}")
            count += 1
            if count % 400 == 0:
                print(f"Verified mounted original SHA-256 {count}", flush=True)
    return count


def progress(args, config):
    rows = []
    for variant in config["variants"]:
        for seed in config["seeds"]:
            name = f"{variant}_s{seed}"
            folder = args.output / "runs" / name
            history = read_json(folder / "history.json") if (folder / "history.json").exists() else []
            rows.append({"run": name, "epochs": len(history), "complete": (folder / "complete.json").exists()})
    return rows


def epoch_reserve(history):
    times = [float(row["seconds"]) for row in history]
    durations = [b-a for a, b in zip([0.] + times[:-1], times)]
    return max(600., 2. * max(durations, default=0.))


def train_until_budget(args, deadline, on_progress=lambda: None, clock=time.monotonic):
    """One verified cache read per session, one restart-safe epoch per call."""
    from src.straw_global_local import runner
    config, _, frames, spec = runner.open_experiment(args, verify_cache=True)
    device = runner.device_for(args)
    args.max_epochs_this_call = 1
    for job in runner.jobs(config):
        folder = args.output / "runs" / job["id"]
        while True:
            if (folder / "complete.json").exists():
                runner.verify_run(args.output, job, spec)
                print(f"SKIP completed {job['id']}", flush=True)
                break
            history = read_json(folder / "history.json") if (folder / "history.json").exists() else []
            if deadline - clock() < epoch_reserve(history):
                print(f"SOFT STOP before {job['id']} epoch {len(history)+1}; checkpoint preserved", flush=True)
                on_progress()
                return False
            runner.train_one(args, config, frames, spec, job, device)
            on_progress()
    return True


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_session(bundle, project, *, input_root=Path("/kaggle/input"), output_root=Path("/kaggle/working"),
                session_hours=10.5, resume_from=None, evaluate_test=False, review_evidence=None,
                benchmark_enabled=True, started=None, device="cuda", raw_root=None, pretrained_file=None):
    if not math.isfinite(session_hours) or not 0 < session_hours <= 10.5:
        raise ValueError("SESSION_HOURS must be in (0, 10.5]; reserve time for snapshot/output saving")
    started = time.monotonic() if started is None else started
    deadline = started + session_hours * 3600
    project, output_root = Path(project).resolve(), Path(output_root).resolve()
    import torch
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before starting; no CPU fallback for the full sweep")
    project.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = materialize_bundle(bundle, project)
    identity = {"bundle": manifest["fingerprint"], "project_root": str(project),
                "config": "configs/straw_global_local.json", "output": OUTPUT_REL.as_posix()}
    identity_path = project / IDENTITY_REL
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("Live session belongs to a different bundle; use a clean workspace")
        print("Continuing verified live workspace; attached resume ZIP will not overwrite it.")
    else:
        resume = find_resume(input_root, resume_from)
        if resume:
            print(f"Restoring trusted previous output: {resume}", flush=True)
            restore_snapshot(resume, project, manifest["fingerprint"])
            if read_json(identity_path) != identity:
                raise ValueError("Restored identity mismatch")
        write_json(identity_path, identity)
    sys.path.insert(0, str(project))
    from src.straw_global_local import runner, data
    torch.set_num_threads(2)
    args = argparse.Namespace(config=project / identity["config"], protocol=project / PROTOCOL_REL,
                              cache=project / CACHE_REL, output=project / OUTPUT_REL, device=device,
                              run_id=None, max_epochs_this_call=1, repeats=30, bootstrap=1000)
    runner.device_for(args)
    status = read_json(project / SESSION_REL) if (project / SESSION_REL).exists() else {"sessions": []}
    session = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "environment": runner.environment(torch.device(device)), "budget_hours": session_hours}
    if status["sessions"]:
        previous = status["sessions"][-1]["environment"]
        session["environment_changes"] = {key: {"previous": previous.get(key), "current": value}
                                          for key, value in session["environment"].items()
                                          if previous.get(key) != value}
    status["sessions"].append(session)
    config = read_json(args.config)
    def update(state=None):
        if state:
            status["state"] = state
        status["runs"] = progress(args, config)
        status["completed_runs"] = sum(row["complete"] for row in status["runs"])
        status["total_runs"] = len(status["runs"])
        session["elapsed_seconds"] = time.monotonic() - started
        write_json(project / SESSION_REL, status)
        write_json(output_root / "straw_session_status.json", status)
    log_path = project / "kaggle_session.log"
    error, dataset_link = None, None
    with log_path.open("a", encoding="utf-8", buffering=1) as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            update("preparing")
            if session.get("environment_changes"):
                print("Environment changed since previous session; exact numerical equivalence across environments is not guaranteed.")
            if config.get("pretrained", True):
                ensure_pretrained(input_root, pretrained_file)
            image_tree, prefix = discover_dataset(project, input_root, raw_root)
            image_root, dataset_link = bind_dataset(project, image_tree, prefix)
            session["dataset_mount"] = str(image_tree)
            print(f"Reading original images directly from Kaggle input: {image_tree}")
            if not (args.protocol / "protocol.json").exists():
                prepare_stable_protocol(project, args.protocol, image_root)
            if review_evidence is not None:
                if (args.output / "freeze.json").exists() and read_json(args.protocol / "review.json") != review_evidence:
                    raise ValueError("Cannot change review after selection freeze")
                if not runner.check_review(review_evidence):
                    raise ValueError("REVIEW_EVIDENCE must contain actual attributed review, not placeholder flags")
                data.write_json(args.protocol / "review.json", review_evidence)
            # A restored cache must never hide a wrong dataset mount that happens
            # to have the same filenames, especially for raw-image benchmarks.
            if (args.cache / "cache.json").exists():
                session["original_images_sha256_verified"] = verify_originals(args.protocol)
            if not (args.cache / "cache.json").exists() or len(read_json(args.cache / "cache.json")["files"]) != manifest["image_count"]:
                data.create_cache(args.protocol, args.cache, workers=4)
                session["original_images_sha256_verified"] = manifest["image_count"]
            update("training")
            all_trained = train_until_budget(args, deadline, update)
            if all_trained and benchmark_enabled:
                for variant in config["variants"]:
                    if deadline - time.monotonic() < 600:
                        break
                    job_id = f"{variant}_s{config['seeds'][0]}"
                    target = args.output / "runs" / job_id / "benchmark.json"
                    if not target.exists():
                        args.run_id = [job_id]
                        runner.benchmark(args)
                args.run_id = None
            test_done = False
            if all_trained and evaluate_test:
                if not runner.check_review(read_json(args.protocol / "review.json")):
                    print("TEST WAITING: provide actual REVIEW_EVIDENCE; training checkpoints are saved.")
                elif deadline - time.monotonic() >= 1800:
                    runner.freeze(args)
                    runner.test(args)
                    test_done = True
                else:
                    print("TEST WAITING: time reserve too small; resume in the next session.")
            runner.report(args)
            benchmarks_done = all((args.output / "runs" / f"{v}_s{config['seeds'][0]}" / "benchmark.json").exists() for v in config["variants"])
            status["test_requested"] = bool(evaluate_test)
            status["test_evaluated_this_session"] = test_done
            update("training_complete" if all_trained else "paused_time_budget")
            status["benchmarks_complete"] = benchmarks_done
            status["next_action"] = ("resume_training" if not all_trained else
                                     "resume_benchmarks" if benchmark_enabled and not benchmarks_done else
                                     "review_and_evaluate_test" if not test_done else "analyze_results")
            update()
        except BaseException as exc:
            error = exc
            status["error"] = f"{type(exc).__name__}: {exc}"
            update("error_checkpoint_preserved")
            traceback.print_exc()
        finally:
            log.flush()
            try:
                make_snapshot(project, output_root / RESUME_NAME, manifest["fingerprint"])
            finally:
                # Do not let Kaggle's Output collector follow an input-data link.
                if dataset_link is not None and dataset_link.is_symlink():
                    dataset_link.unlink()
    print(f"{status['completed_runs']}/{status['total_runs']} runs complete. State: {status['state']}")
    print(f"Next session: attach this notebook's saved Output or {RESUME_NAME}; keep the same input bundle.")
    if error is not None:
        raise error
    return status
