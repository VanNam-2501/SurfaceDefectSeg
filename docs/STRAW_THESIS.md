# Khóa luận: kiểm tra khuyết tật ống hút trên hộp đồ uống

> **Hướng dẫn hạ tầng lịch sử:** kế hoạch nghiên cứu chính hiện tại là
> [STRAW_RESEARCH_PLAN.md](STRAW_RESEARCH_PLAN.md). Cấu hình hai tầng và 72 run
> dưới đây chưa triển khai mô hình toàn cục–cục bộ trong kế hoạch mới.

**Đánh giá lại:** đọc [STRAW_RESEARCH_REVIEW.md](STRAW_RESEARCH_REVIEW.md) trước
khi chạy toàn bộ. Bộ 72 run dưới đây là kế hoạch hạ tầng dự kiến; chưa chứng
minh chất lượng phương pháp. Ưu tiên pilot giả thuyết và sửa đối chứng trước.

Tên đề tài gợi ý: **Phân loại chọn lọc hai tầng cho kiểm tra khuyết tật ống hút
trên bao bì hộp đồ uống và đánh giá khả năng tổng quát giữa các thương hiệu**.

Mục tiêu là đánh giá bằng thực nghiệm liệu hệ thống hai tầng giúp cân bằng bỏ
sót lỗi, loại nhầm sản phẩm tốt và khối lượng kiểm tra thủ công tốt hơn mô hình
5 lớp hay không. Không mặc định hai tầng phải thắng. Không coi nhiều dashboard
hoặc thêm nhiều kiến trúc là bằng chứng về đóng góp khoa học.

## Câu hỏi nghiên cứu và đóng góp có thể bảo vệ

1. RQ1: Hai tầng Good/Defect rồi chẩn đoán 4 lỗi có tốt hơn phân loại trực tiếp
   5 lớp trong cùng điều kiện dữ liệu, backbone, số seed và ngân sách huấn luyện?
2. RQ2: Temperature scaling, hiệu chỉnh prior và các ngưỡng từ chối ảnh hưởng
   thế nào đến FNR, macro F1, tỷ lệ REVIEW và chi phí giả định?
3. RQ3: Kết quả thay đổi thế nào khi giữ riêng một thương hiệu hoặc làm ảnh mờ,
   tối, nhiễu, nén JPEG? Ảnh suy giảm tổng hợp là stress test, không phải dữ liệu
   dây chuyền sản xuất thực tế hoặc bằng chứng phát hiện OOD.
4. RQ4: Độ phân giải 384, augmentation bảo toàn ảnh và trọng số lớp có đem lại
   cải thiện đáng kể so với chi phí tài nguyên?

Đóng góp dự kiến: quy trình dữ liệu có nguồn gốc và nhóm ảnh liên quan; so sánh
công bằng giữa các phương pháp; phân tích risk–coverage và tổng quát liên thương
hiệu; bộ mã và demo có thể tái lập. Chỉ viết kết luận sau khi có kết quả thật.

## Quy mô thí nghiệm

| Nhóm | Thiết lập | Số run huấn luyện |
|---|---|---:|
| Chính | ResNet18, EfficientNet-B0, Swin-T × seed 42/43/44 × tầng 1/tầng 2/full5, 224 px | 27 |
| Độ phân giải | ResNet18, 384 px × 3 seed × 3 mô hình | 9 |
| Augmentation | ResNet18, không augmentation × 3 seed × 3 mô hình | 9 |
| Trọng số lớp | ResNet18 tầng 1, trọng số Defect=3 × 3 seed; tái sử dụng tầng 2/full5 gốc | 3 |
| Tổng quát thương hiệu | 8 LOBO fold × ResNet18 × seed 42 × 3 mô hình | 24 |
| Tổng đề xuất | 48 run IID mở rộng + 24 run LOBO | **72** |

Ưu tiên 27 run chính trước. LOBO một seed là giới hạn ngân sách, không được mô
tả thành LOBO ba seed. Muốn ba seed trên mọi fold cần mở rộng cấu hình riêng
và tăng thêm 48 run. VMamba không nằm trong bộ chính vì runtime/weights chưa
được chuẩn hóa; mã đã bỏ việc âm thầm thay VMamba bằng ConvNeXt.

Mỗi nhóm 3 checkpoint được so sánh bằng **7 phương án quyết định**:

- Full5 thông thường, không từ chối.
- Full5 chọn lọc có temperature scaling và ngưỡng REVIEW: đối chứng công bằng
  cho lợi ích của cơ chế từ chối.
- Cascade cứng, không từ chối.
- Cascade triage đầy đủ.
- Triage không hiệu chỉnh prior.
- Triage không temperature scaling.
- Triage không confidence/margin guard tầng 2.

Các ablation bỏ một thành phần được hiệu chỉnh lại trên cùng Validation, cùng
grid đăng ký trước. Không dùng Test để chọn ablation hoặc ngưỡng tốt nhất.

## Dữ liệu và protocol

Audit local ngày 2026-10-18: 2.000 ảnh, 8 thương hiệu, 5 lớp; không có bản trùng
RGB hoàn toàn, 76 cặp ứng viên dHash khoảng cách <=2. Các thành phần liên thông
của ứng viên được giữ cùng tập theo cách bảo thủ: 1.943 nhóm. Điều này không xác
nhận các cặp đều là ảnh trùng và không phát hiện đầy đủ quan hệ theo phiên chụp.
Cần rà soát trực quan và bổ sung capture/session ID nếu thu thập được.

Protocol nội bộ đang tạo:

| Tập | Vai trò | Ảnh local |
|---|---|---:|
| Train | Học trọng số, augmentation chỉ tại đây | 1.201 |
| Validation | Chọn checkpoint, ngưỡng/policy và phương án triển khai | 300 |
| Calibration | Fit temperature; không chọn checkpoint | 200 |
| Test | Đánh giá các so sánh đã đăng ký sau khi freeze | 299 |

Tỷ lệ mục tiêu 60/15/10/15; số thực có thể lệch vì không tách nhóm. Stratification
theo thương hiệu × lớp, group theo nội dung và ứng viên tương tự. Các fold LOBO
giữ toàn bộ một thương hiệu cho Test; phần còn lại tách 70/15/15 cho
Train/Validation/Calibration, loại bản liên quan từ thương hiệu khác nếu nhóm
đó đã thuộc Test. Mỗi fold fit lại temperature và policy riêng; không dùng
threshold từ toàn bộ dữ liệu để đánh giá thương hiệu giữ lại.

**Dữ liệu này đã được dùng trong quá trình phát triển trước đây.** Bộ chia mới
là benchmark nội bộ, không trở thành tập kiểm chứng bên ngoài chưa từng thấy.
Muốn có xác nhận mạnh hơn, thu thêm ảnh ở phiên chụp, thiết bị, lô hàng hoặc
thương hiệu mới; khóa tập đó trước khi xem dự đoán và không dùng để phát triển.
Không nhân bản/augment ảnh rồi gọi đó là tăng số mẫu độc lập.

`prepare` đọc ảnh gốc để audit, giữ danh tính/nhóm ảnh và tạo cache PNG 384×384
bảo toàn toàn bộ ảnh để giảm thời gian đọc ảnh gốc 3000×3000 mỗi epoch. Cả train
và inference dùng cùng cache; thí nghiệm 224 resize tiếp từ cache. Đây là bước
tiền xử lý chung, không phải dữ liệu mới. Các manifest local `data_v2` hiện tại
vẫn trỏ ảnh gốc; nếu dùng cache, tạo protocol/output mới và không trộn hai kết quả.

## Thiết lập huấn luyện và quyết định

Cấu hình duy nhất: `configs/straw_thesis.json`. Tối đa 40 epoch, early stopping
patience 10, AdamW, encoder LR 1e-4, classifier LR 1e-3, weight decay 1e-4,
label smoothing 0,05 cho cả ba nhiệm vụ. Effective batch 32: CNN batch 32,
Swin batch 8 × accumulation 4; 384 px giảm batch một nửa và tăng accumulation.
BatchNorm với accumulation không tương đương một batch lớn; phải công khai giới
hạn này khi so sánh. Cùng effective batch cũng không đồng nghĩa cùng FLOPs.

Tầng 1 chọn checkpoint bằng recall Defect tại FPR<=5% trên Validation. Tầng 2
và full5 chọn macro recall trên Validation. Dùng ImageNet pretrained cho mọi
run chính; smoke dùng random initialization và 1 epoch, không được đưa vào bảng
khóa luận. Thời gian mỗi run, actual best epoch và environment được lưu riêng.

Policy dùng grid hữu hạn trong config. Fit temperature bằng Calibration,
chọn ngưỡng trên Validation; ưu tiên thỏa FNR<=1% và tỷ lệ REVIEW triển khai
giả định <=30%, rồi giảm chi phí. Nếu không có nghiệm, ghi **infeasible**;
không âm thầm nới mục tiêu hoặc tuyên bố an toàn. Policy mỗi seed được lưu riêng.
Lựa chọn họ mô hình dựa trên trung bình Validation của các seed; checkpoint
triển khai dùng seed nhỏ nhất trong họ, không chọn seed theo Test.

Prior triển khai 5% và chi phí FN/REVIEW/FP/sai loại lỗi = 100/2/5/5 là **giả
định phân tích**. Hiệu chỉnh prior dựa vào giả thiết label shift; lớp có trọng
số/label smoothing có thể làm giả thiết này không chính xác. Vì vậy ablation
không prior và phân tích độ nhạy là bắt buộc. Chi phí có đơn vị tương đối,
không tự quy đổi thành tiền hoặc hiệu quả kinh tế nhà máy.

## Chỉ số và bảng cần có

- Accuracy, macro F1 trên mọi ảnh; REVIEW được tính là chưa phân loại đúng.
- Recall từng lớp, macro recall lỗi, confusion matrix 5×6 (thêm cột REVIEW).
- FNR = lỗi được PASS / toàn bộ ảnh lỗi; FPR = ảnh Good bị DEFECT / toàn bộ Good.
- Coverage, selective risk chỉ trên ảnh được tự động quyết định, tỷ lệ REVIEW
  toàn bộ và có điều kiện theo Good/Defect, tỷ lệ REVIEW khi giả định prior 5%.
- AUROC, average precision; NLL, ECE 15 bins, multiclass Brier trước/sau hiệu
  chỉnh. ECE trên Calibration dùng để fit không phải kiểm chứng độc lập;
  diagnostics Validation/Test được lưu riêng.
- Wilson 95% CI cho FNR/FPR và paired bootstrap 1.000 lần theo nhóm, stratify
  theo lớp, cho chênh lệch triage − full5. Wilson theo ảnh là mô tả nhị thức;
  bootstrap theo nhóm phản ánh một phần phụ thuộc giữa ảnh.
- Trung bình ± độ lệch chuẩn qua 3 seed, báo cáo riêng với CI lấy mẫu. LOBO
  báo cáo từng thương hiệu và trung bình giữa fold; không gọi SD giữa thương
  hiệu là SD giữa seed.
- Ảnh clean + 4 họ corruption × 3 mức = 13 điều kiện. Các tham số đã cố định,
  nhiễu xác định theo sample ID; giữ nguyên policy, không retune trên Test.
- Độ nhạy chi phí/prior: prior 1/5/10/20%, FN cost 20/100/500, giữ policy cố định.
- Latency thực đo p50/p95, batch 1, warmup 10 lần, đo 100 lần; số tham số và
  peak CUDA memory. Benchmark forward không bao gồm đọc ảnh/preprocess/API.
  Không lấy tổng latency hai model nhân một tỷ lệ tầng 2 tùy ý rồi gọi là thực đo.

**FNR=0 trên 240 ảnh lỗi không chứng minh FNR thực dưới 1%.** Wilson hai phía
95% khi không có lỗi cần ít nhất 381 mẫu lỗi độc lập để cận trên <=1%; vẫn
không bảo đảm điều kiện sản xuất mới hoặc sửa chữa được việc đã xem Test.

## Chạy tại local

```powershell
# Kiểm tra logic/protocol, không tải pretrained weights
.\verify_straw.ps1

# Đã có protocol local tại artifacts/straw_thesis/data_v2/iid.
# Với dữ liệu hoặc máy mới, tạo protocol riêng; output phải chưa có dữ liệu:
.\run_straw_thesis.ps1 -Action prepare -Output artifacts/straw_thesis/data_cached_v1

# Xem ngân sách. Core và extended phải dùng output riêng.
.\run_straw_thesis.ps1 -Action plan
.\run_straw_thesis.ps1 -Action plan -Suite extended -Output artifacts/straw_thesis/extended_v1

# Chạy chính; có thể chia run bằng -RunId lấy từ plan.json
.\run_straw_thesis.ps1 -Action train
.\run_straw_thesis.ps1 -Action calibrate
.\run_straw_thesis.ps1 -Action freeze
.\run_straw_thesis.ps1 -Action test
.\run_straw_thesis.ps1 -Action benchmark
.\run_straw_thesis.ps1 -Action report
```

Chạy lại `train` tiếp tục từ `last.pt` nếu gián đoạn, bỏ qua run đã hoàn thành
và khớp hash. Lưu model/optimizer/scheduler/AMP/best epoch/patience/RNG cùng
checkpoint. Config, mã hoặc protocol thay đổi phải dùng output mới. Không tái
sử dụng checkpoint smoke chỉ vì có file `best.pt`. Giữ nguyên toàn bộ source
bundle khi chạy các shard; cache và kết quả có hash để kiểm tra danh tính.

Smoke toàn luồng:

```powershell
.\run_straw_thesis.ps1 -Action smoke-data -Output artifacts/straw_thesis/smoke_data_v1
$ProtocolPath = 'artifacts/straw_thesis/smoke_data_v1'
$ResultPath = 'artifacts/straw_thesis/smoke_v2'
foreach ($Step in @('train','calibrate','freeze','test','report')) {
  .\run_straw_thesis.ps1 -Action $Step -Suite smoke -Protocol $ProtocolPath -Output $ResultPath
  if ($LASTEXITCODE -ne 0) { throw "Failed: $Step" }
}
```

LOBO protocol:

```powershell
.\run_straw_thesis.ps1 -Action lobo-data -Output artifacts/straw_thesis/lobo_data_v1
# Ví dụ một fold; lặp lại với đủ 8 thư mục protocol.
.\run_straw_thesis.ps1 -Action train -Protocol artifacts/straw_thesis/lobo_data_v1/choco -Output artifacts/straw_thesis/lobo_choco_v1
```

Mỗi fold tiếp tục `calibrate`, `freeze`, `test`, `report` với cùng Protocol và
Output. Fold mặc định chỉ chạy ResNet18 seed 42. Báo cáo từng fold không tự động
trở thành một kết quả tổng hợp LOBO; dùng script tổng hợp ở `scripts/research`.

## Kaggle và thời gian 4–6 tuần

Đóng gói bằng `python scripts/research/package_straw_kaggle.py`, upload ZIP source
làm private Kaggle Dataset và gắn dataset ảnh vào notebook
`experiments/notebooks/straw_thesis_kaggle.ipynb`. Bật GPU, cho phép tải pretrained
weights nếu cần; giữ cặp torch/torchvision sẵn có. Đo thời gian một nhóm 3 run
trước khi ước lượng toàn bộ ngân sách. Giới hạn Kaggle và GPU có thể thay đổi;
kiểm tra trong tài khoản, không giả định luôn đủ quota cho 72 run.

1. Tuần 1: chốt đề tài với giảng viên; rà 76 cặp tương tự, nguồn gốc dữ liệu,
   nhãn và split; chạy smoke; đăng ký protocol/config; pilot một nhóm 3 run.
2. Tuần 2: hoàn thành 27 run chính; kiểm tra mất cân bằng REVIEW, calibration
   và các trade-off chỉ bằng Validation; khóa mọi quyết định chính.
3. Tuần 3: chạy phần mở rộng và 8 fold LOBO; khóa policy trong từng fold trước
   Test; lưu log lỗi và thất bại thực nghiệm, không chỉ giữ run tốt.
4. Tuần 4: bảng, bootstrap, robustness, phân tích lỗi, benchmark; tích hợp model
   đã khóa vào demo; viết chương kết quả, quay video và chuẩn bị bảo vệ.
5. Nếu còn tuần 5–6: thu một tập ảnh ngoài độc lập, đánh giá xác nhận, sửa báo
   cáo và diễn tập. Nếu thiếu GPU, ưu tiên core + LOBO hơn tăng kiến trúc.

Demo `hierarchical_app.py` hiện dùng đường dẫn checkpoint/config cũ. Không tự
động thay demo bằng model smoke hoặc policy mới chưa có kết quả. Sau khi chọn
candidate thật cần tích hợp đúng preprocessing, class order và policy đã khóa,
kiểm tra thiếu checkpoint phải trả unavailable. Giải thích occlusion là phân
tích độ nhạy, không phải chứng minh quan hệ nhân quả; EWMA là thống kê cảnh báo,
không tự chứng minh chẩn đoán nguyên nhân máy móc.

## Tổ chức luận văn và bằng chứng

Chương 1: vấn đề và phạm vi; Chương 2: nền tảng, transfer learning, calibration,
selective prediction; Chương 3: dữ liệu và phương pháp; Chương 4: thực nghiệm;
Chương 5: demo, hạn chế và kết luận. Mỗi bảng phải dẫn experiment fingerprint,
split hash, config, actual epoch, seed và checkpoint. Không sao chép bảng Test
cũ vào bộ kết quả mới. Giữ báo cáo thất bại/infeasible thay vì loại để đẹp số.

Thư mục output chứa `experiment.json`, `plan.json`, `runs/`, `cache/`,
`selection/`, `freeze.json`, `test/`, `benchmarks/`, `report/`. HTML báo cáo ghi
rõ SMOKE, INCOMPLETE hoặc COMPLETED INTERNAL BENCHMARK; không suy ra trạng thái
hoàn thành từ việc có ảnh biểu đồ.

Nguồn phương pháp:

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017:
  https://proceedings.mlr.press/v70/guo17a.html
- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, 2017:
  https://arxiv.org/abs/1705.08500

Pipeline ở đây áp dụng các ý tưởng temperature scaling và đánh đổi risk–coverage;
không tuyên bố đã triển khai hoặc kế thừa bảo đảm lý thuyết của các bài báo.
