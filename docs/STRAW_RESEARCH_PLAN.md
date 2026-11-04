# Kế hoạch nghiên cứu chính: phân loại khuyết tật ống hút bằng CNN

Cập nhật 18-10-2026 theo phạm vi người dùng: chỉ dataset hiện có; đầu ra là
sản phẩm bình thường hoặc bị lỗi gì. Đây là đề cương để kiểm chứng, chưa phải
phương pháp đã chứng minh hiệu quả hoặc xác nhận tính mới.

**Tổng hợp tài liệu và ưu tiên mới:** [STRAW_RESEARCH_SYNTHESIS.md](STRAW_RESEARCH_SYNTHESIS.md).
Phạm vi triển khai đã chốt: global-local và đối chứng vị trí B6, **không
chưng cất**. Các mục KD trong bản tổng hợp là lịch sử phương án mở rộng.
Code B0–B6 hiện nằm trong src/straw_global_local. Xem
[runbook](STRAW_GLOBAL_LOCAL_RUNBOOK.md) và
[điều kiện đủ bằng chứng kết luận](STRAW_COMPLETION_REQUIREMENTS.md).

Tài liệu này là điểm đọc chính, thay thế định hướng trung tâm về chuyển miền
thương hiệu, MixStyle và hệ thống PASS/REVIEW trong các kế hoạch trước.
Các mã nguồn và kết quả cũ được giữ để tham khảo; chúng chưa triển khai mô hình
toàn cục–cục bộ dưới đây. Tiêu chí chính thức của trường/giảng viên chưa được
cung cấp, nên chưa thể xác nhận đạt chuẩn nộp khóa luận.

## 1. Bài toán, đầu ra và phạm vi kết luận

- Dataset: 2.000 ảnh, 8 thương hiệu, 5 lớp, mỗi ô thương hiệu × lớp có 50 ảnh.
- Nhãn: good, bent, loose, missing, pierced. Cần xác minh định nghĩa gốc của
  pierced và tiêu chí phân biệt good/loose/pierced; không tự dịch thành lỗi
  thủng thân ống hút khi chưa có bằng chứng.
- Mô hình chính: phân loại đơn nhãn 5 lớp. Nếu dữ liệu có ảnh đồng thời nhiều
  lỗi, cần ghi quy tắc gán nhãn và xử lý mơ hồ; không tự chuyển sang đa nhãn.
- Quyết định mặc định: y_hat = argmax p(y|x). Good là bình thường, bốn nhãn
  còn lại là sản phẩm lỗi. Trạng thái tốt/lỗi suy từ cùng quyết định 5 lớp.
- Điểm lỗi liên tục s = 1 - p(good) dùng cho ROC/PR. Không mặc định ngưỡng 0,5
  trên s cho cùng quyết định với argmax 5 lớp. Nếu thử ngưỡng riêng, phải khai
  báo thành chính sách phụ, chọn trên Validation và đánh giá riêng.
- Thương hiệu là metadata, không là nhãn đầu ra, không cần nhập khi suy luận.
  LOBO không thuộc thí nghiệm bắt buộc. Kết luận chính nói về phân loại ảnh
  trong phạm vi dataset và phép chia đã kiểm tra.
- Không có nhãn vị trí độc lập: không báo IoU/mAP định vị hoặc độ chính xác
  phân đoạn từ attention/pseudo-mask. Không hứa phát hiện mọi loại lỗi mới.

## 2. Đánh giá mức sẵn sàng

| Nội dung | Đánh giá | Việc cần bổ sung |
|---|---|---|
| Nhiệm vụ ứng dụng | Rõ | Chuẩn hóa định nghĩa nhãn |
| Dữ liệu và công cụ chia tập | Đã có nền tảng | Rà nhóm ảnh, ghi nguồn và điều kiện sử dụng |
| Câu hỏi nghiên cứu | Có thể kiểm chứng | Khóa chỉ số chính và phép so sánh |
| Cơ sở chọn toàn cục–cục bộ | Pilot ngắn chưa cho thấy lợi ích | Kiểm tra đủ ngân sách, không mặc định kiến trúc sẽ thắng |
| Tính mới | Chưa xác nhận | Bảng related work có khác biệt cụ thể |
| Mô hình mới | Đã triển khai B0–B6 | Xác thực pipeline và huấn luyện đủ theo protocol |
| Hiệu quả và chi phí | Đã có pilot/benchmark một seed | Thực nghiệm xác nhận nhiều seed, CI và phân tích |

Smoke test, một bước backward hoặc số lượng run không phải bằng chứng hiệu quả.
Khóa luận có thể báo kết quả âm có phân tích; không khẳng định trước mô hình
đề xuất sẽ thắng baseline.

## 3. Câu hỏi và giả thuyết

Tên dự kiến: **Nghiên cứu kết hợp đặc trưng toàn cục và cục bộ trong mạng CNN
cho phân loại khuyết tật ống hút.** Vị trí là thành phần khảo sát có ablation,
không mặc định là một đóng góp dương.

RQ1: Độ phân giải ảnh đầu vào ảnh hưởng thế nào đến phân loại từng loại lỗi?
H1: Thu nhỏ ảnh có thể làm mất dấu hiệu cần thiết của một số lớp. Đối chứng
là cùng CNN với ảnh toàn cảnh 224 và 512, không suy ra H1 đúng chỉ từ ảnh mẫu.

RQ2: Thông tin toàn ảnh và các vùng ảnh có bổ sung nhau không?
H2: Ngữ cảnh giúp nhận biết hình dạng/vắng mặt, còn các vùng ảnh giữ được chi
tiết. Đối chứng gồm toàn ảnh, vùng ảnh đơn thuần và kết hợp trung bình.

RQ3: Attention phụ thuộc ngữ cảnh toàn ảnh có giúp hơn tổng hợp trung bình
hoặc attention chỉ nhìn từng vùng không?
H3: Ngữ cảnh có thể giúp chọn bằng chứng cục bộ phù hợp. Đối chiếu thêm B6/B5
để đo vai trò của thông tin vị trí crop, cùng tập view, encoder và classifier.

Đánh giá thực dụng xuyên suốt: lợi ích của từng thiết kế có đáng chi phí
tiền xử lý, xử lý nhiều view và suy luận không?

Kết luận có điều kiện: nếu RQ1 không cho thấy hạn chế rõ và các pilot local
không có tín hiệu hữu ích, không tiếp tục thêm module để buộc có cải thiện.
Nếu B3 tốt bằng B5 với chi phí thấp hơn, lựa chọn cuối có thể là B3.

## 4. Chuẩn hóa dữ liệu và phép chia

1. Đọc nguồn dataset, ghi tác giả/phiên bản/điều kiện sử dụng. Bổ sung bảng
   định nghĩa 5 nhãn, ví dụ đúng và trường hợp khó phân biệt. Nếu có hai người
   rà nhãn, báo mức đồng thuận; nếu chỉ một người, ghi đúng thực tế.
2. Rà 76 cặp dHash gần nhau và nhóm liên quan, ghi quyết định giữ/gộp/tách.
   1.943 nhóm hiện có là kết quả heuristic; không gọi đó là 1.943 đối tượng
   hoặc phiên chụp đã được xác nhận độc lập. Không có metadata phiên chụp thì
   nêu giới hạn này. Nếu cần sửa nhóm, tạo phiên bản split mới trước train.
3. Giữ cả nhóm trong một tập. Stratify theo lớp; metadata thương hiệu có thể
   hỗ trợ giữ phân bố, không biến thành mục tiêu dự đoán.
4. Protocol mới tại artifacts/straw_global_local/data_v1 đã gộp Train và
   Calibration cũ thành 1.401 Train, giữ 300 Val và 299 Test. Manifest cũ được
   giữ nguyên. Nếu rà nhãn/nhóm gây thay đổi, xuất protocol mới trước train;
   mọi phương án dùng cùng fingerprint. Không trộn hai protocol hoặc dùng
   checkpoint/output của hướng cũ cho bảng kết quả mới.
5. Khóa Test trong quá trình phát triển. Dataset có lịch sử được dùng trước;
   mô tả benchmark nội bộ, không gọi là test ngoài mới hoàn toàn.
6. 5 lớp cân bằng nhưng gộp nhị phân có 400 Good và 1.600 Defect. Không dùng
   accuracy nhị phân đơn lẻ hoặc tự thêm trọng số mất cân bằng cho 5 lớp.

## 5. Thiết kế mô hình tối thiểu để kiểm chứng

### 5.1. Tạo view và tiền xử lý

- Global: giữ toàn khung hình, resize/letterbox nhất quán; xử lý EXIF trước.
- Local v1: 4 cửa sổ chồng lấn, kích thước mỗi cửa sổ bằng 60% chiều rộng và
  chiều cao ảnh; bốn góc phủ toàn khung hình. Mỗi crop resize về 224 × 224.
  Đây là lưới phủ ảnh, không được gọi là ROI ống hút đã được định vị.
- Cắt từ ảnh gốc trước khi resize. Cache 384 × 384 do runner cũ tạo không đủ
  cho giả thuyết giữ chi tiết gốc; không phóng cache đó rồi gọi là high-res.
  Nếu cần cache, tạo cache view có cấu hình và nguồn ảnh được fingerprint.
- Một biến đổi hình học toàn ảnh, nếu dùng, thực hiện trước khi tạo tất cả
  view để giữ quan hệ vị trí nhất quán. Augmentation nhẹ cần rà giữ nhãn.
  Random crop/erase làm mất ống hút có thể đổi ngữ nghĩa good/missing; không
  tự áp dụng chỉ vì đó là augmentation phổ biến.
- Không gán nhãn lỗi của toàn ảnh cho từng patch: patch có thể chỉ chứa nền.
  Loss chính áp dụng vào dự đoán của cả ảnh sau tổng hợp.

### 5.2. Encoder, tổng hợp và hàm mục tiêu

Dùng ResNet18 ImageNet pretrained với encoder dùng chung cho các view:

$$g=f_\theta(G(x)),\quad h_k=f_\theta(P_k(x)),\quad k=1,\ldots,K.$$

Attention có điều kiện toàn ảnh:

$$s_k=v^T\tanh(W_hh_k+W_gg+b),\qquad
a_k=\frac{\exp(s_k)}{\sum_j\exp(s_j)},\qquad
z_l=\sum_k a_kh_k.$$

$$p_\theta(y\mid x)=\operatorname{softmax}(c_\phi([g,z_l])),\qquad
L=-\frac1N\sum_i\log p_\theta(y_i\mid x_i).$$

Loss đầu tiên chỉ là CE ở cấp ảnh. Trọng số lớp, JS, contrastive, focal loss
không được thêm đồng thời với kiến trúc, vì sẽ không tách được nguyên nhân.
Không áp giả định MIL nhị phân rằng mọi ảnh lỗi phải có ít nhất một patch lỗi:
Missing là bằng chứng vắng mặt, còn Bent có thể cần cấu trúc trải nhiều patch.

### 5.3. Nội dung toán học có thể phân tích

- Trong mô hình đơn giản của feature map, tín hiệu tăng Delta ở m/M vị trí
  làm GAP thay đổi (m/M)Delta. Đây là lập luận về khả năng pha loãng, không
  chứng minh CNN thực tế bỏ sót lỗi hoặc crop chắc chắn tốt hơn.
- Với h cố định theo score, đạo hàm dz_l/ds_j = a_j(h_j-z_l). Kết hợp với
  gradient CE cho thấy việc ưu tiên patch phụ thuộc khả năng giảm loss.
  Cơ chế này vẫn có thể ưu tiên họa tiết nền; không bảo đảm định vị đúng.
- Mean pooling và attention không có tọa độ là bất biến với hoán vị tập patch.
  Vì vậy nhánh local không trực tiếp giữ thứ tự hình học giữa các patch.
  Global cung cấp ngữ cảnh nhưng không bảo đảm khôi phục mọi quan hệ. Nếu
  khảo sát vấn đề này, B6 thêm tọa độ crop chuẩn hóa vào score attention,
  s_k = v^T tanh(W_h h_k + W_g g + W_r phi(r_k) + b). Phải so với B5;
  không mặc định tọa độ giúp ích hoặc khôi phục mọi quan hệ hình học.
- Phân tích chi phí: global 224 + 4 local 224 xử lý 250.880 pixel/view-set;
  global 512 xử lý 262.144 pixel. Đây là đối chứng gần nhau về lượng pixel,
  không khẳng định FLOPs hoặc latency bằng nhau; phải đo trên cùng phần cứng.
  Crop 60% resize 224 có mật độ theo chiều tương đương global khoảng 373,
  nên không mặc định giữ chi tiết tốt hơn global 512 dù có nhiều view.

## 6. Ma trận thí nghiệm chính

| ID | Đầu vào / tổng hợp | Câu hỏi |
|---|---|---|
| B0 | Global 224 | Baseline |
| B1 | Global 512 | Tăng độ phân giải đơn thuần |
| B2 | 4 local 224, mean, không global | Local có đủ không? |
| B3 | Global + local mean, ghép đặc trưng | Ngữ cảnh và chi tiết có bổ sung? |
| B4 | Như B3, attention chỉ dùng h_k | Học trọng số có ích hơn mean? |
| B5 | Như B3, attention dùng h_k và g | Lợi ích riêng của điều kiện global? |
| B6 | B5 + tọa độ crop chuẩn hóa | Thông tin vị trí có ích thêm không? |

Pilot một seed trên Train/Val trước. Sau khi chốt protocol huấn luyện, mục
tiêu là 7 cấu hình × 3 seed = 21 run xác nhận, không gồm tuning và pilot.
Con số này là ngân sách dự kiến, không phải thước đo chất lượng khóa luận.
Nếu không đủ compute, ưu tiên B0/B1/B3/B5 × 3 seed; khi thiếu B4/B6 phải bỏ
tuyên bố lợi ích riêng của conditioning/vị trí tương ứng. Ghi rõ thu hẹp.

Không chưng cất trong phạm vi hiện hành. Mô hình được chọn bằng Validation
là mô hình tích hợp trực tiếp vào ứng dụng sau khi đánh giá xác nhận.

Khảo sát phụ theo thứ tự ưu tiên, chỉ khi còn ngân sách:
1. Kích thước hoặc số lượng crop, đổi một yếu tố mỗi lần.
2. Baseline mạnh nhất và phương án chọn trên EfficientNet-B0, cùng số seed.
3. Blur/tối/JPEG ở mức cố định; là stress test nhân tạo.

LOBO, hai tầng, REVIEW, prior giả định và kiểm soát FNR 1% không thuộc lõi.
MixStyle là prototype lịch sử; nếu dùng làm đối chứng cần ngân sách riêng.

## 7. Huấn luyện và so sánh công bằng

- Cùng split, pretrained nguồn, augmentation và quy tắc checkpoint.
- Đơn vị batch là ảnh gốc (bag), không coi 5 view là 5 ảnh độc lập. Dùng
  gradient accumulation nếu cần giữ effective image batch bằng nhau.
- Số bước cập nhật và giới hạn epoch theo một quy tắc chung. Cùng ngân sách
  tìm learning rate cho mỗi phương án; ghi cả run thất bại và hyperparameter.
- Khóa chính sách BatchNorm: nhiều view làm thay đổi running statistics. Có
  thể thử trên baseline việc giữ running stats pretrained rồi áp cùng chính
  sách cho mọi phương án; affine có học hay không phải ghi rõ. Không dùng
  chính sách khác tùy phương án mà bỏ qua trong báo cáo.
- RNG dữ liệu tách khỏi RNG module, cùng seed/data order khi có thể. Shared
  encoder giảm số tham số nhưng không giảm số lần xử lý view; báo cả chi phí.
- Một lựa chọn khởi đầu để pilot: AdamW, encoder LR 1e-4/head 1e-3, giới hạn
  40 epoch, patience 10; đây là cấu hình thử, không phải cấu hình tối ưu đã
  chứng minh. Trần tìm kiếm LR phải định trước và đối xứng giữa phương án.
- Chọn checkpoint bằng macro-F1 Val, tie-break bằng Val CE rồi epoch sớm hơn.
  Chọn family/config bằng kết quả Val trung bình qua seed. Test dùng để báo
  kết quả của các cấu hình đã khóa; không sửa mô hình rồi thử lại Test liên tục.
- Pilot đo thời gian/epoch, VRAM, tốc độ đọc ảnh gốc trên Kaggle GPU thực tế
  trước khi ước tính tổng lịch. Không hứa hoàn thành theo số epoch lý thuyết.

## 8. Chỉ số, độ bất định và tiêu chí kết luận

Chính: macro-F1 5 lớp. Bổ sung accuracy, recall/precision/F1 từng lớp và ma
trận nhầm lẫn. Good/Defect suy từ argmax cho cùng quyết định đầu ra:

$$FNR=\frac{\#\{y\ne Good,\hat y=Good\}}{\#\{y\ne Good\}},\qquad
FPR=\frac{\#\{y=Good,\hat y\ne Good\}}{\#\{y=Good\}}.$$

Sai Bent thành Loose là lỗi chẩn đoán, không là bỏ sót sản phẩm lỗi. Báo riêng.
Không chỉ tính F1 trên các ảnh mô hình đã phát hiện đúng là lỗi vì dễ che các
trường hợp bỏ sót. Báo raw count cùng tỷ lệ, tránh độ chính xác giả từ mẫu ít.

- Mean ± SD qua 3 seed thể hiện độ nhạy huấn luyện, không là CI lấy mẫu.
- Paired bootstrap theo nhóm ảnh trên cùng test prediction để đánh giá chênh
  lệch, giữ các ảnh trong nhóm đi cùng nhau. Báo CI theo seed, hoặc dùng quy
  tắc bootstrap trung bình các metric qua seed đã chốt; không gộp 3 seed thành
  3 lần số ảnh độc lập. CI vẫn phụ thuộc tính hợp lý của nhóm đã nhận diện.
- Chọn một so sánh chính trước Test: global-local family được chọn trên Val
  với baseline toàn ảnh được chọn trên Val từ B0/B1. B5–B4 và B6–B5 là đối
  chiếu cơ chế. Nếu làm
  nhiều kiểm định thì khai báo hiệu chỉnh hoặc chỉ diễn giải mô tả thăm dò.
- Không đặt mức tăng 1–2% như chuẩn khoa học tự động. Trước Test, chốt mức
  cải thiện có ý nghĩa thực tế và giới hạn chi phí theo yêu cầu ứng dụng nếu
  có; nếu chưa có, báo đường đánh đổi hiệu năng/chi phí thay vì chuẩn công nghiệp.
- Chênh lệch nhỏ, CI chứa 0 hoặc hiệu ứng không ổn định: kết luận chưa có bằng
  chứng ưu thế. B5 thua B1 cũng là kết quả cần phân tích, không giấu baseline.
- Thời gian end-to-end gồm đọc/tiền xử lý/cắt vùng/forward; báo riêng forward,
  batch=1, warmup, đồng bộ GPU, p50/p95, GPU và phiên bản phần mềm.

## 9. Related work và giới hạn đóng góp

| Nguồn | Liên quan | Việc cần làm thêm |
|---|---|---|
| Ilse et al., ICML 2018, Attention-based Deep MIL | Attention tổng hợp tập instance, học nhãn bag | Đọc công thức/protocol; phân biệt MIL nhị phân với bài 5 lớp này |
| Hu et al., WS-DAN, 2019 | Attention và khai thác vùng phân biệt từ nhãn ảnh | Đối chiếu crop, attention, cách train và test |
| Gumapac, Sciforum abstract/poster 15-10-2026 | Cùng dataset 2.000 ảnh, fine-tuning DenseNet121/Xception/InceptionV3 | Đọc poster/protocol; không so trực tiếp accuracy nếu split khác |

Nguồn gốc:
- https://proceedings.mlr.press/v80/ilse18a.html
- https://arxiv.org/abs/1901.09891
- https://sciforum.net/paper/34988

Đây là danh sách khởi đầu, không phải rà soát tính mới đầy đủ. Cần bổ sung
nghiên cứu sát về context-conditioned attention/global-local classification,
ghi bảng giả định, nhãn cần dùng, chi phí và khác biệt chính xác. Nếu claim
so với phương pháp đã công bố, cần baseline tái triển khai cùng protocol hoặc
giới hạn claim tương ứng; các ablation nội bộ chưa thay thế đối chứng đó.
Không gọi ghép CNN + attention tự thân là thuật toán hoàn toàn mới hoặc SOTA.

## 10. Mốc thực hiện và sản phẩm cần có

| Mốc | Sản phẩm | Điều kiện qua mốc |
|---|---|---|
| A: dữ liệu và đề cương | Nhãn, nguồn/license, group audit, split mới, related work | Không còn mơ hồ nghiêm trọng về mục tiêu và split |
| B: baseline/pilot | Learning curves, Val errors, profiling B0/B1 và local | Xác định được giả thuyết cần tiếp tục hoặc bác bỏ |
| C: thực nghiệm chính | Trainer, config, checkpoint, 21 run lõi dự kiến, prediction | Có đối chứng trả lời từng RQ, protocol khóa |
| D: kết quả và luận văn | Bảng/CI/chi phí, thất bại, demo và giới hạn | Mỗi claim truy được về dữ liệu thực nghiệm |

Với ít nhất 4 tuần, dự trù mỗi mốc khoảng một tuần nhưng điều chỉnh sau pilot
GPU; không dành toàn bộ thời gian cuối cho viết luận văn. Demo chỉ bắt đầu
tích hợp model cuối khi preprocessing/class mapping/checkpoint đã khóa.

Notebook mới phải hỗ trợ ảnh gốc/cached views, batch theo ảnh, encoder chung,
CE ở cấp ảnh, cấu hình B0–B6, resume, provenance và xuất logits/sample IDs.
Các kiểm thử cần thiết: tách nhóm không giao nhau, view không dùng nhãn,
bag loss/backward đúng, inference không cần nhãn/thương hiệu, lớp đầu ra nhất
quán, tái lập/resume và phép tính chỉ số. Kiểm thử code không là kết quả mô hình.

Luận văn: (1) Bài toán và phạm vi; (2) Cơ sở/related work; (3) Dữ liệu và
protocol; (4) Phương pháp và phân tích; (5) Thí nghiệm/kết quả/chi phí;
(6) Kết luận, giới hạn và ứng dụng minh họa.

## 11. Trạng thái tại thời điểm viết

Đã triển khai src/straw_global_local: B0–B6, view cache từ ảnh gốc, trainer
resume, chọn checkpoint bằng Val, chốt lựa chọn trước Test, benchmark và báo
cáo readiness. Đã tạo protocol ba tập mới và notebook Kaggle riêng. 22 kiểm
thử đã đạt; xem [hồ sơ kiểm chứng](STRAW_GLOBAL_LOCAL_VERIFICATION.md) để biết
kết quả smoke/pilot thực tế.

Chưa hoàn thành thực nghiệm xác nhận 21 run, rà định nghĩa nhãn/nhóm/quyền sử
dụng và novelty, hoặc phân tích kết quả cuối. Code hoàn tất không có nghĩa
đã đủ bằng chứng để viết kết luận hiệu quả. Mô tả độ sẵn sàng dựa trên
STRAW_COMPLETION_REQUIREMENTS.md và artifact thực, không tự đánh dấu "đạt".
