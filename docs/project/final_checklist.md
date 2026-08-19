# Final run checklist

> **Lưu ý hồ sơ cuối:** đây là checklist kế hoạch ban đầu. Scope kết quả đã
> chốt là E0, E2, E3, E4, E5, E7 và E8; E1/E6/E9/E10/E11 không nằm trong lần
> báo cáo hiện tại. Cấu hình batch thực tế phải lấy từ checkpoint và được ghi
> tại `FINAL_EXPERIMENT_CONFIG.md`, không lấy con số effective batch = 4 ở tài
> liệu protocol cũ.

Before each main run:

- [ ] `/content/TTTN` is the working directory
- [ ] `python check_protocol.py` prints PASS
- [ ] same frozen split hashes are used
- [ ] patch size 512, stride 256
- [ ] augmentation = `photometric`
- [ ] seed = 42 for main table
- [ ] BCE(all) + Dice(positive only)
- [ ] early stopping enabled
- [ ] result directory is new / not overwritten accidentally
- [ ] GPU name recorded in `environment.json`

For VMamba additionally:

- [ ] T4 / sm75 runtime matches the prebuilt wheel
- [ ] `selective_scan_cuda` imports
- [ ] `WITH_SELECTIVESCAN_MAMBA=True`
- [ ] `test_vmamba_runtime.py` passes

After training:

- [ ] `checkpoints/best.pt` exists
- [ ] learning loss + Dice curves exist
- [ ] monitor images exist
- [ ] `training_summary.json` records best epoch and times

Before Test:

- [ ] run Validation evaluator first
- [ ] `selected_threshold.json` exists
- [ ] threshold selected only from Val

After Test:

- [ ] `main_metrics.csv`
- [ ] ROC / PR / confusion matrix
- [ ] defect-size table
- [ ] defect-group table
- [ ] multi-region table
- [ ] qualitative failures
- [ ] timing/VRAM recorded

After all three models:

- [ ] run `compare_models.py`
- [ ] run U-Net E1 resize baseline
- [ ] decide whether E10/E11 are actually justified by error analysis
- [ ] optionally run seeds 123 and 2026, then `compare_seeds.py`
