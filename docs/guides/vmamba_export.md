# Xuất VMamba từ Kaggle để làm báo cáo 3 model

Checkpoint VMamba đã có, không cần train lại. File
`export_probability_cache.py` đã nằm trong project/gói Kaggle cuối, nên không
cần upload helper riêng. Chạy một cell mới trong notebook Kaggle và thay 3
đường dẫn nếu notebook của bạn dùng tên khác.

```python
import sys
from pathlib import Path
import shutil, zipfile

# PROJECT là thư mục chứa train_on_kaggle.py, vmamba_t.py và export_probability_cache.py.
PROJECT = Path('/kaggle/working/Aluminum_Surface_Defect_Segmentation')
DATASET = Path('/kaggle/input/<project-dataset>/data/3cad_ani')
CHECKPOINT = Path('/kaggle/working/results/vmamba_b8_main_seed42/checkpoints/best.pt')
OUT = Path('/kaggle/working/vmamba_predictions')

# Copy helper vừa upload vào code project đang chạy được. Nếu file .py đã có
# trong PROJECT thì cell này không làm thay đổi gì.
if not PROJECT.joinpath('export_probability_cache.py').is_file():
    helper = next(Path('/kaggle/input').rglob('export_probability_cache.py'), None)
    helper_zip = next(Path('/kaggle/input').rglob('export_vmamba_kaggle_helper.zip'), None)
    if helper:
        shutil.copy2(helper, PROJECT / 'export_probability_cache.py')
    elif helper_zip:
        with zipfile.ZipFile(helper_zip) as zf:
            zf.extract('export_probability_cache.py', PROJECT)

assert PROJECT.joinpath('export_probability_cache.py').is_file(), PROJECT
assert CHECKPOINT.is_file(), CHECKPOINT
assert DATASET.is_dir(), DATASET

%cd {PROJECT}
!{sys.executable} export_probability_cache.py \
  --model vmamba \
  --checkpoint {CHECKPOINT} \
  --dataset-root {DATASET} \
  --output-root {OUT} \
  --splits val test \
  --tile-size 512 --stride 256 --tile-batch-size 16
```

Lệnh có thể chạy lại nếu bị ngắt: ảnh PNG đã tồn tại được bỏ qua. Khi xong, nén thư mục để tải về:

```python
import shutil
shutil.make_archive('/kaggle/working/vmamba_predictions', 'zip', '/kaggle/working/vmamba_predictions')
print('/kaggle/working/vmamba_predictions.zip')
```

Tải `vmamba_predictions.zip` từ Output của Kaggle. Giải nén sao cho cấu trúc local là:

```text
E:\Project\TTTN\artifacts\experiments\decision\predictions\vmamba\val\probability\<image_id>.png
E:\Project\TTTN\artifacts\experiments\decision\predictions\vmamba\test\probability\<image_id>.png
```

Không để thừa một tầng `vmamba_predictions` bên trong thư mục đó. Sau khi giải nén, chạy ở PowerShell local:

```powershell
cd E:\Project\TTTN
.\run_three_model_experiments.ps1 -Action Check
.\run_three_model_experiments.ps1
```

Kết quả sẽ nằm trong
`E:\Project\TTTN\artifacts\experiments\decision\three_model_experiment_report\tables`.
