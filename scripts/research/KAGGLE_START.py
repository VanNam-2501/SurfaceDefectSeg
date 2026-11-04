"""Entry point included at the root of the all-in-one Kaggle bundle."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import zipfile

_started = time.monotonic()
_bundle = Path(globals().get("BUNDLE_PATH") or Path(__file__).resolve().parent)
_project = Path(globals().get("PROJECT_ROOT", "/kaggle/working/straw-global-local"))
_module_name = "scripts/research/kaggle_session.py"
if _bundle.is_dir():
    _manifest = json.loads((_bundle / "bundle_manifest.json").read_text())
    _code = (_bundle / _module_name).read_bytes()
else:
    with zipfile.ZipFile(_bundle) as _archive:
        _manifest = json.loads(_archive.read("bundle_manifest.json"))
        _code = _archive.read(_module_name)
if hashlib.sha256(_code).hexdigest() != _manifest["files"][_module_name]["sha256"]:
    raise ValueError("Session runner checksum mismatch; upload the original bundle again")
_module_path = _project / _module_name
_module_path.parent.mkdir(parents=True, exist_ok=True)
_module_path.write_bytes(_code)
_spec = importlib.util.spec_from_file_location("straw_kaggle_session", _module_path)
_session = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_session)
_session.run_session(
    _bundle, _project,
    input_root=Path(globals().get("INPUT_ROOT", "/kaggle/input")),
    output_root=Path(globals().get("OUTPUT_ROOT", "/kaggle/working")),
    session_hours=globals().get("SESSION_HOURS", 10.5),
    resume_from=globals().get("RESUME_FROM", None),
    evaluate_test=globals().get("EVALUATE_TEST", False),
    review_evidence=globals().get("REVIEW_EVIDENCE", None),
    benchmark_enabled=globals().get("BENCHMARK", True),
    raw_root=globals().get("RAW_ROOT", None),
    pretrained_file=globals().get("PRETRAINED_FILE", None),
    started=_started,
)
