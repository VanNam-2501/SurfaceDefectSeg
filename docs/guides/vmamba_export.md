# Xuất VMamba từ Kaggle để làm báo cáo 3 model

Checkpoint VMamba đã có, không cần train lại. File
`export_probability_cache.py` đã nằm trong project/gói Kaggle cuối, nên không
cần upload helper riêng. Chạy một cell mới trong notebook Kaggle và thay 3
đường dẫn nếu notebook của bạn dùng tên khác.

```python
import sys
from pathlib import Path
import shutil, zipfile

# PROJECT là gốc package, có các thư mục src/ và scripts/.
PROJECT = Path('/kaggle/working/threecad_ani_project')
DATASET = Path('/kaggle/input/<project-dataset>/data/3cad_ani')
CHECKPOINT = Path('/kaggle/working/results/vmamba_b8_main_seed42/checkpoints/best.pt')
OUT = Path('/kaggle/working/vmamba_predictions')

EXPORTER = PROJECT / 'scripts/experiments/export_probability_cache.py'
assert EXPORTER.is_file(), EXPORTER
assert CHECKPOINT.is_file(), CHECKPOINT
assert DATASET.is_dir(), DATASET

%cd {PROJECT}
!{sys.executable} {EXPORTER} \
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
.\scripts\experiments\run_three_model_experiments.ps1 -Action Check
.\scripts\experiments\run_three_model_experiments.ps1
```

Kết quả sẽ nằm trong
`E:\Project\TTTN\artifacts\experiments\decision\three_model_experiment_report\tables`.
