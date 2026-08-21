# Quy trình thí nghiệm — 3CAD Aluminum New iPad

Tài liệu này quy định quy trình cố định cho các lần chạy cuối của khóa luận tốt
nghiệp. Sau khi bắt đầu huấn luyện chính thức, không được tự ý thay đổi tập
Train/Validation/Test, hàm mất mát, quy tắc đánh giá hoặc phép tăng cường dữ liệu
chính cho riêng một kiến trúc.

## Quy trình chính đã cố định

Áp dụng cùng một quy trình cho U-Net/ResNet18, SegFormer-B0 và VMamba-T:

- sử dụng cố định `train.csv / val.csv / test.csv`;
- sử dụng ảnh nguồn ở độ phân giải gốc;
- Train: lấy các patch `512x512` trực tiếp trong quá trình huấn luyện;
- ảnh Defect → crop dương có nhận biết GT; ảnh Good → crop âm ngẫu nhiên;
- Validation/Test: suy luận toàn ảnh bằng tile `512`, stride `256`, lấy trung
  bình tại các vùng chồng lấn;
- Validation/Test tuyệt đối không dùng GT để chọn vị trí crop;
- chuẩn hóa theo ImageNet;
- tăng cường dữ liệu Train mặc định: chỉ biến đổi quang học nhẹ:
  - độ sáng ±10%;
  - độ tương phản ±10%;
  - gamma ±10%;
- lật ngang/dọc và xoay 90 độ chỉ được bật tùy chọn
  (`photometric_geometric`) nếu đã xác nhận hướng của sản phẩm không mang ý
  nghĩa vật lý;
- hàm mất mát: `0.5 * BCE(tất cả mẫu) + 0.5 * Dice(chỉ mẫu có GT dương)`;
- AdamW, weight decay `1e-4`;
- learning rate encoder `1e-5`, decoder `1e-4`;
- tối đa `50` epoch;
- effective batch size bằng `4`:
  - U-Net/SegFormer: batch 2 × gradient accumulation 2;
  - VMamba: batch 1 × gradient accumulation 4;
- bật AMP khi chạy CUDA;
- gradient clipping `1.0`;
- dùng ReduceLROnPlateau theo Positive Dice@0.5 trên Validation toàn ảnh;
- early stopping patience `8`, min delta `1e-4`;
- chọn checkpoint tốt nhất bằng **Validation Positive Dice@0.5**, tuyệt đối
  không dùng Train loss;
- threshold cuối cùng chỉ được chọn trên Validation;
- tuyệt đối không dùng Test để tinh chỉnh.

### Vì sao chỉ tính Dice trên mẫu dương?

BCE được tính trên cả patch Good và Defect để các vùng bình thường giúp model
hạn chế false positive. Dice chỉ được tính trên những mẫu có pixel GT dương,
nhờ đó mask Good toàn số 0 không trở thành một bài toán tối ưu độ chồng lấn.

### Công bằng về pretraining

So sánh chính E2 chỉ sử dụng classification pretraining:

- encoder U-Net: ResNet18 pretrained trên ImageNet;
- SegFormer-B0: `nvidia/mit-b0` pretrained trên ImageNet; binary decode head
  được khởi tạo mới;
- VMamba-T: VMamba-T s2l5 chính thức pretrained trên ImageNet; binary FPN head
  được khởi tạo mới.

Không được dùng checkpoint SegFormer đã fine-tune trên ADE20K từ bước smoke test
cho thí nghiệm E2.

## Bằng chứng tự động được lưu trong quá trình huấn luyện

Mỗi lần chạy phải tạo ra:

- `config.json`;
- `environment.json`;
- `model_info.json`;
- `split_snapshot/{train,val,test}.csv`;
- `split_snapshot/sha256.json`;
- `training_history.csv`;
- `training_summary.json`;
- `checkpoints/best.pt`;
- `checkpoints/last.pt`;
- `checkpoints/interrupt.pt` nếu quá trình huấn luyện bị gián đoạn;
- `curves/learning_curve_loss.png`;
- `curves/learning_curve_dice.png`;
- `curves/loss_components.png`;
- `curves/learning_rate.png`;
- `curves/epoch_time.png`;
- `curves/vram.png`;
- các ví dụ Validation toàn ảnh cố định trong `monitor/epoch_XXX/`.

Lịch sử Validation bao gồm một giá trị loss chẩn đoán trên ảnh toàn phần đã
được ghép:

- Val BCE trên mọi ảnh;
- Val soft Dice loss chỉ trên ảnh Defect;
- `Val loss = 0.5*Val BCE + 0.5*Val positive Dice loss`.

Val loss chỉ dùng để theo dõi đường học. Early stopping vẫn sử dụng Positive
Dice@0.5 trên Validation toàn ảnh.

# Các thí nghiệm

## E0 — Kiểm tra tính toàn vẹn kỹ thuật / smoke test — đã hoàn thành

Chỉ cần kiểm tra lại khi môi trường chạy thay đổi:

- ảnh và mask thẳng hàng;
- patch đầu ra có ảnh `[3,512,512]`, mask `[1,512,512]`;
- model trả đầu ra `[B,1,512,512]`;
- forward/backward cho kết quả hữu hạn;
- VMamba: import được `selective_scan_cuda` và
  `WITH_SELECTIVESCAN_MAMBA=True`.

Mini-overfit chỉ là bước kiểm tra kỹ thuật trước khi chạy, không phải thí nghiệm
hiệu năng để báo cáo.

## E1 — Nghiên cứu phương pháp tiền xử lý — ngoài phạm vi bàn giao cuối

E1 không được chạy trong bộ kết quả cuối theo phạm vi đã chốt. Cấu hình đang dùng
là huấn luyện patch trên ảnh độ phân giải gốc và suy luận toàn ảnh bằng sliding
window; không có nhánh resize đối chứng trong bảng bàn giao.

## E2 — So sánh kiến trúc chính

- U-Net + ResNet18;
- SegFormer-B0;
- VMamba-T s2l5.

Báo cáo:

- Image AUROC;
- Image AUPRC;
- Precision / Recall / Specificity / F1;
- FNR / FPR;
- Positive Dice / Positive IoU;
- Pixel AUPRC / Recall / Precision;
- Region Recall;
- tổng số tham số / số tham số trainable;
- peak VRAM khi Train/Test;
- thời gian huấn luyện;
- số tile/giây;
- độ trễ toàn ảnh mean / P50 / P95;
- tổng thời gian đánh giá Test.

Điểm image-level được định nghĩa là `max(probability map toàn ảnh)`.

Quá trình Test tự động lưu:

- `figures/image_roc_curve.png` và CSV;
- `figures/image_pr_curve.png` và CSV;
- `figures/confusion_matrix.png` và CSV.

## E3 — Phân tích theo kích thước khuyết tật

Ngưỡng Tiny/Small/Medium/Large được suy ra từ **các connected component chỉ
thuộc tập TRAIN** bằng Q1/Q2/Q3.

Báo cáo:

- region recall theo từng component và từng nhóm kích thước;
- image recall theo kích thước component GT nhỏ nhất;
- Positive Dice/IoU theo kích thước component GT nhỏ nhất.

## E4 — Phân tích theo nhóm khuyết tật

Nhóm lỗi chỉ là metadata; model vẫn thực hiện phân đoạn nhị phân:

- bump;
- bruise;
- scratches;
- knife mark;
- Multiple-defects.

Báo cáo số ảnh, image recall, Positive Dice và IoU. Những nhóm có ít hơn `10`
ảnh Test phải được tự động đánh dấu là kết quả mô tả do cỡ mẫu nhỏ.

## E5 — Phân tích ảnh có nhiều vùng lỗi

Các nhóm số lượng component được suy ra chỉ từ thống kê Train.

Báo cáo số ảnh/số vùng, image recall, region recall, Positive Dice và IoU.

## E6 — Hiệu quả tính toán

Mọi phép đo thời gian phải được thực hiện trên cùng GPU và cùng runtime.

Báo cáo:

- tổng số tham số / số tham số trainable;
- epoch tốt nhất;
- tổng thời gian huấn luyện;
- thời gian chỉ dành cho Train;
- thời gian Validation trong quá trình huấn luyện;
- peak VRAM khi Train;
- peak VRAM khi Test;
- thông lượng model-forward theo tile/giây;
- độ trễ toàn ảnh mean/P50/P95;
- tổng thời gian đánh giá Test.

**Quy tắc đo thời gian:** trước khi đo Validation/Test cuối, chạy 5 batch warm-up
cho model. Không tính thời gian warm-up vào latency hoặc throughput được báo cáo.

Không được đặt các số liệu thời gian thu từ những loại GPU khác nhau vào cùng
bảng so sánh E6 chính.

## E7 — Độ nhạy theo threshold

Validation tự động lưu:

- `threshold_scan.csv`;
- `threshold_selection.png`;
- `selected_threshold.json`.

Quy tắc:

1. quét từ `0.05..0.95`, bước `0.01`;
2. giữ các threshold có Image FNR ≤ 10%;
3. chọn threshold có Positive Dice cao nhất;
4. nếu bằng nhau → chọn FPR thấp hơn;
5. cố định threshold;
6. áp dụng threshold đó đúng một lần lên Test.

Nếu không có threshold nào thỏa ràng buộc FNR, code phải ghi rõ việc sử dụng
quy tắc fallback.

## E8 — Phân tích lỗi định tính

Quá trình đánh giá cuối lưu:

- các ảnh Good có điểm dự đoán cao nhất — ứng viên false positive;
- các ảnh Defect có Dice thấp nhất;
- ảnh Original / GT / probability / binary overlay.

Dùng các ví dụ này để thảo luận về scratches, điểm lỗi rất nhỏ, sai lệch tại
biên và texture lặp lại.

## E9 — Khả năng tái lập — ngoài phạm vi bàn giao cuối

Không chạy thí nghiệm nhiều seed trong phạm vi hiện tại. Script cũ được lưu ở
`archive/review_candidates/2026-08-21/out_of_scope_experiments/` để có thể phục
hồi nếu phạm vi báo cáo thay đổi.

## E10 — Ablation hard-negative sampling — ngoài phạm vi bàn giao cuối

Không chạy ablation hard-negative trong bộ kết quả cuối. Sampler chính vẫn giữ:
ảnh Defect dùng positive crop có nhận biết GT, ảnh Good dùng negative crop ngẫu
nhiên.

## E11 — Hậu xử lý component nhỏ — ngoài phạm vi bàn giao cuối

Không báo cáo E11 như một ablation độc lập. Logic quyết định giảm dương tính giả
được đánh giá trong pipeline policy/hybrid riêng, luôn chọn tham số trên
Validation rồi khóa trước khi áp dụng lên Test.

# Định nghĩa đánh giá toàn ảnh

- Positive Dice/IoU: macro average chỉ trên các ảnh Defect;
- dự đoán image-level: `max(probability_map) >= selected_threshold`;
- Region Recall: một GT component được coi là phát hiện nếu có ít nhất một pixel
  dự đoán dương chồng lên nó; tương đương với max probability bên trong
  component ≥ threshold;
- Pixel AUPRC: xấp xỉ bằng histogram toàn cục với 4096 bins để tránh lưu xác
  suất của toàn bộ pixel độ phân giải gốc trong RAM;
- chọn threshold: chỉ dùng Validation;
- Test: dùng checkpoint cuối đã cố định và threshold đã cố định, không tinh
  chỉnh thêm.

# Thứ tự chạy được khuyến nghị

1. `check_protocol.py`;
2. huấn luyện U-Net chính;
3. chọn threshold cho U-Net trên Validation;
4. đánh giá U-Net trên Test;
5. chạy SegFormer chính + Validation + Test;
6. thiết lập/kiểm tra runtime VMamba;
7. chạy VMamba chính + Validation + Test;
8. chạy `compare_models.py`;
9. xem lại lỗi E8 và data-audit; không dùng các kết quả audit để chỉnh hồi tố Test.

# Cấu trúc kết quả cuối bắt buộc

```text
results/
├── unet_r18/
│   └── main_seed42/
├── segformer_b0/
│   └── main_seed42/
├── vmamba_t_s2l5/
│   └── main_seed42/
└── comparison/
    ├── main_metrics.csv
    ├── defect_size_metrics.csv
    ├── defect_group_metrics.csv
    ├── multi_region_metrics.csv
    ├── efficiency.csv
    ├── e1_preprocessing.csv
    ├── architecture_comparison.png
    └── efficiency_comparison.png
```
