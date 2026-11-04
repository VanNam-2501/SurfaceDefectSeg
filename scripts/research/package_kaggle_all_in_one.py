"""Build one CODE-ONLY ZIP for all 21 experiments; images stay in Kaggle Input."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
from package_global_local import source_files
from kaggle_session import canonical_hash, safe_name, sha256

BOOTSTRAP = '''# Kaggle: GPU ON. Run using Save Version -> Save & Run All.
SESSION_HOURS = 10.5  # Stop between epochs, leaving time to save outputs.
RESUME_FROM = None  # Auto-detect previous straw_resume.zip; "" starts fresh.
RAW_ROOT = None  # Auto-find dataset; or set the Kaggle image directory explicitly.
PRETRAINED_FILE = None  # Internet ON for initial download, or attach official .pth.
EVALUATE_TEST = False  # Enable only after actual data/literature review.
REVIEW_EVIDENCE = None
BENCHMARK = True

from pathlib import Path
import zipfile
sources = [p for p in Path('/kaggle/input').rglob('KAGGLE_START.py')
           if (p.parent/'bundle_manifest.json').is_file()]
if sources:
    assert len(sources) == 1, 'Attach exactly one all-in-one source dataset.'
    entry = sources[0]
    code = entry.read_text(encoding='utf-8')
    BUNDLE_PATH = str(entry.parent)
else:
    archives = list(Path('/kaggle/input').rglob('straw_kaggle_code.zip'))
    assert len(archives) == 1, 'Attach straw_kaggle_code.zip as an input dataset.'
    BUNDLE_PATH = str(archives[0])
    entry = archives[0]/'KAGGLE_START.py'
    with zipfile.ZipFile(archives[0]) as archive:
        code = archive.read('KAGGLE_START.py').decode('utf-8')
exec(compile(code, str(entry), 'exec'), dict(globals(), __name__='__main__', __file__=str(entry)))
'''


def notebook():
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": [
                {"cell_type": "markdown", "id": "instructions", "metadata": {}, "source": [
                    "# CNN toàn cục–cục bộ: 21 lượt, tự lưu và tiếp tục\n",
                    "Upload **straw_kaggle_code.zip** thành Kaggle Dataset riêng tư; Add Input cùng dataset ảnh gốc đang có trên Kaggle. Gói chỉ chứa code/config/split/notebook, không có dataset hoặc weight. Bật GPU và Internet để tải weight ResNet18; giữ torch/torchvision của Kaggle.\n",
                    "RAW_ROOT=None tự tìm thư mục chứa đúng 2.000 đường dẫn ảnh trong manifest, không phụ thuộc slug dataset. Nếu có nhiều bản dataset, đặt RAW_ROOT thủ công. Đường dẫn CSV ổn định qua các phiên bằng alias, không copy ảnh vào Output.\n",
                    "Chọn **Save Version → Save & Run All** để chạy nền. Mỗi epoch lưu checkpoint; gần ngân sách giờ sẽ dừng bình thường và xuất `/kaggle/working/straw_resume.zip`. Phiên sau Add Input Output của phiên trước; chạy lại cùng notebook để tiếp tục.\n",
                    "Không sửa code/config/bundle/PROJECT_ROOT khi resume. Nếu treo máy hoặc Kaggle cưỡng bức ngắt trước lúc Output được lưu, file trong working có thể mất; checkpoint không tự đồng bộ lên cloud. Lưu phiên bằng Run All và để chương trình dừng sớm.\n",
                    "Mặc định tự train đủ B0–B6 × seed 42/43/44 và benchmark trên Validation. Test chỉ chạy khi EVALUATE_TEST=True và REVIEW_EVIDENCE là kết quả rà thực tế; không tự ghi cờ giả.\n"]},
                {"cell_type": "code", "id": "run-session", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": BOOTSTRAP.splitlines(keepends=True)},
                {"cell_type": "code", "id": "download-output", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": ["from IPython.display import display, FileLink\n",
                            "display(FileLink('/kaggle/working/straw_resume.zip'))\n",
                            "print(Path('/kaggle/working/straw_session_status.json').read_text())\n"]}
            ]}


def embedded_notebook(archive_path):
    """A single importable .ipynb: source ZIP embedded, external images only."""
    payload = archive_path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    value = notebook()
    value["metadata"]["straw_code_bundle_sha256"] = checksum
    value["cells"][0]["source"] = [
        "# 21 thí nghiệm CNN — một notebook chứa toàn bộ code\n",
        "**File này không chứa dataset hoặc weight.** Import thẳng vào Kaggle, Add Input dataset ảnh gốc, bật GPU và Internet để tải weight chính thức. Không cần upload thêm ZIP code.\n",
        "Chọn **Save Version → Save & Run All**. Mặc định B0–B6 × seed 42/43/44, 40 epoch/patience10. Dataset được đọc trực tiếp từ Input; RAW_ROOT=None tự dò hoặc đặt đường dẫn thủ công.\n",
        "Mỗi epoch lưu checkpoint. Gần ngân sách 10,5 giờ, chương trình dừng và xuất straw_resume.zip. Phiên sau gắn Output của phiên trước làm Input và chạy lại cùng notebook; không train lại lượt đã xong. Phải để Kaggle lưu Output thì mới có tiến độ cho phiên mới.\n",
        "Code/config/split/docs/tests được nhúng dưới dạng ZIP base64 để giữ một file duy nhất, có SHA-256 kiểm tra. Cell thứ hai có thể dài; không cần sửa payload. Bản source ZIP tách riêng được giao để đọc và kiểm tra dễ hơn.\n",
        "EVALUATE_TEST mặc định False; chỉ bật sau review thực tế, không dùng Test để tuning. Notebook đã kiểm thử cục bộ, chưa chạy trên Kaggle.\n",
    ]
    options = BOOTSTRAP.split("from pathlib import Path", 1)[0]
    code = options + "from pathlib import Path\nimport base64, hashlib, zipfile\n"
    code += "_encoded_source = " + repr(base64.b64encode(payload).decode("ascii")) + "\n"
    code += "_source_bytes = base64.b64decode(_encoded_source)\n"
    code += f"assert hashlib.sha256(_source_bytes).hexdigest() == {checksum!r}, 'Embedded source checksum mismatch'\n"
    code += "_source_zip = Path('/kaggle/working/straw_kaggle_code.zip')\n_source_zip.parent.mkdir(parents=True, exist_ok=True)\n_source_zip.write_bytes(_source_bytes)\n"
    code += "BUNDLE_PATH = str(_source_zip)\nwith zipfile.ZipFile(_source_zip) as _archive:\n    _entry = _archive.read('KAGGLE_START.py').decode('utf-8')\n"
    code += "exec(compile(_entry, str(_source_zip/'KAGGLE_START.py'), 'exec'), dict(globals(), __name__='__main__', __file__=str(_source_zip/'KAGGLE_START.py')))\n"
    value["cells"][1]["source"] = code.splitlines(keepends=True)
    return value


def build(output):
    files = {p.relative_to(ROOT).as_posix(): p for p in source_files()}
    files.update({
        "scripts/research/kaggle_session.py": ROOT / "scripts/research/kaggle_session.py",
        "scripts/research/package_kaggle_all_in_one.py": ROOT / "scripts/research/package_kaggle_all_in_one.py",
        "tests/test_kaggle_session.py": ROOT / "tests/test_kaggle_session.py",
        "KAGGLE_START.py": ROOT / "scripts/research/KAGGLE_START.py",
        "scripts/research/KAGGLE_START.py": ROOT / "scripts/research/KAGGLE_START.py",
        "README_KAGGLE.md": ROOT / "docs/KAGGLE_ALL_IN_ONE.md",
        "docs/KAGGLE_ALL_IN_ONE.md": ROOT / "docs/KAGGLE_ALL_IN_ONE.md",
        "START_HERE.ipynb": ROOT / "experiments/notebooks/straw_global_local_all_in_one_kaggle.ipynb",
        "experiments/notebooks/straw_global_local_all_in_one_kaggle.ipynb": ROOT / "experiments/notebooks/straw_global_local_all_in_one_kaggle.ipynb",
        "KAGGLE_RUN_CELL.txt": ROOT / "artifacts/packages/KAGGLE_RUN_CELL.txt",
    })
    for name, path in files.items():
        safe_name(name)
        if not path.is_file():
            raise FileNotFoundError(path)
    records = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.zip")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for index, (name, path) in enumerate(sorted(files.items()), 1):
                value, size = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        value.update(chunk)
                        size += len(chunk)
                checksum = value.hexdigest()
                records[name] = {"bytes": size, "sha256": checksum}
                if index % 200 == 0:
                    print(f"Packed and hashed {index}/{len(files)}", flush=True)
            manifest = {"schema": 1, "kind": "straw-kaggle-all-in-one", "files": records,
                        "image_count": 2000, "run_count": 21, "code_only": True,
                        "configuration": "configs/straw_global_local.json",
                        "data_note": "NO IMAGES/WEIGHTS INCLUDED. Attach original Kaggle dataset separately; frozen CSV split metadata is included."}
            manifest["fingerprint"] = canonical_hash(manifest)
            archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {"file": output.name, "bytes": output.stat().st_size, "sha256": sha256(output),
                "bundle_fingerprint": manifest["fingerprint"], "files": len(records), "included_images": 0, "runs": 21}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/packages/straw_kaggle_code.zip")
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    target = ROOT / "experiments/notebooks/straw_global_local_all_in_one_kaggle.ipynb"
    target.write_text(json.dumps(notebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cell = ROOT / "artifacts/packages/KAGGLE_RUN_CELL.txt"
    cell.parent.mkdir(parents=True, exist_ok=True)
    cell.write_text(BOOTSTRAP, encoding="utf-8")
    if not args.assets_only:
        build(args.output)
        standalone = args.output.parent / "STRAW_KAGGLE_ALL_EXPERIMENTS.ipynb"
        standalone.write_text(json.dumps(embedded_notebook(args.output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Self-contained notebook: {standalone} ({standalone.stat().st_size:,} bytes; no images/weights)")


if __name__ == "__main__":
    main()
