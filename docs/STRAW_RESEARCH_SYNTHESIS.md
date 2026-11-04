# Tổng hợp nghiên cứu và định hướng ưu tiên cho khóa luận ống hút

> **Phạm vi hiện hành đã chốt không chưng cất.** Phần teacher/student dưới
> đây lưu phương án đã khảo sát, không thuộc kế hoạch triển khai hiện tại.
> Xem [kế hoạch chính](STRAW_RESEARCH_PLAN.md),
> [runbook](STRAW_GLOBAL_LOCAL_RUNBOOK.md) và
> [điều kiện đủ bằng chứng](STRAW_COMPLETION_REQUIREMENTS.md).

Ngày rà soát: 18-10-2026. Phạm vi: dataset hiện có, CNN, nhãn ảnh 5 lớp,
Kaggle GPU, thời gian ít nhất 4 tuần. Đầu ra ứng dụng: sản phẩm bình thường
hoặc bị lỗi gì. Không bổ sung dataset, không coi thương hiệu là mục tiêu.

Đây là rà soát có mục tiêu và đề xuất thiết kế, không phải systematic review
đầy đủ hoặc chứng nhận tính mới. Nguồn gồm bài gốc, trang hội nghị và mã/tài
liệu đã có trong workspace. Chưa chạy thực nghiệm mới; thứ tự ưu tiên dưới
đây là đánh giá nghiên cứu, không phải xếp hạng hiệu năng đo được.

## 1. Kết luận lựa chọn

Trục nghiên cứu ưu tiên: **học kết hợp bằng chứng toàn ảnh và vùng chi tiết có
thông tin vị trí; sau khi có teacher hữu ích, khảo sát chưng cất sang CNN chỉ
xử lý một ảnh toàn cảnh.**

Hai mục tiêu liên kết:
1. Nhận biết đúng lỗi: tránh mất chi tiết, vẫn giữ ngữ cảnh và hình dạng dài.
2. Chi phí suy luận: khảo sát chuyển lợi ích của mô hình nhiều view sang một
   CNN một lượt. Teacher nhiều view chỉ cần khi train student.

Lựa chọn này nâng chiều sâu của hướng global-local đã có, không ghép đồng thời
GeM, BlurPool, SAM, contrastive, focal và domain adaptation. Các thành phần
trên có thể làm đối chứng/phương án dự phòng theo bằng chứng cụ thể.

Đóng góp lớn không đồng nghĩa module nhiều. Cần chỉ ra một hạn chế, giải pháp,
đối chứng đủ mạnh, lợi ích thực và giới hạn. Chưa có cơ sở hứa tăng 5%, 10%
hoặc vượt phương pháp đã công bố; chưa có baseline đầy đủ cùng protocol.

## 2. Ưu tiên theo tác động và khả năng kiểm chứng

| Ưu tiên | Hạng mục | Tác động kỳ vọng / cơ sở | Vai trò đóng góp | Điều kiện |
|---|---|---|---|---|
| Bắt buộc | Nhãn rõ, chia nhóm, fine-tuning đúng, baseline đủ mạnh | Sai dữ liệu/train có thể lấn át mọi cải tiến model | Nền tảng độ tin cậy, không tự là thuật toán mới | Làm đầu tiên |
| 1 | Giữ chi tiết gốc + toàn cảnh | Bài toán vừa có hình dạng/vắng mặt vừa có dấu hiệu cục bộ | Giả thuyết biểu diễn chính | Pilot độ phân giải và toàn ảnh/local |
| 2 | Tổng hợp có ngữ cảnh và thông tin vị trí | Bag pooling thuần không mô tả trực tiếp vị trí giữa các crop | Thành phần ứng viên, cần ablation | Chỉ giữ nếu Val cho thấy lợi ích ổn định |
| 3 | Chưng cất nhiều view -> một ảnh | Nhiều view tăng chi phí; KD có nền tảng chuyển tri thức | Nghiên cứu hiệu năng–chi phí | Teacher hữu ích, đủ compute và thời gian |
| Bắt buộc | Đối chứng ngoài + CI + lỗi/chi phí | Giúp đánh giá mức đóng góp thực | Đóng góp thực nghiệm và tái lập | Mọi claim có prediction/artifact |
| Phụ | GeM / anti-aliasing / giảm stride | Có thể giúp pooling hoặc ổn định, thay đổi nhỏ và có tiền lệ | Đối chứng rẻ hoặc phân tích cơ chế | Không đưa tất cả vào kiến trúc chính |
| Sau cùng | SAM/contrastive/tăng nhiều backbone | Có thể hữu ích nhưng mở thêm trục tuning | Không ưu tiên trước hạn chế đã xác định | Chỉ khi lỗi/learning curve tạo động cơ |

## 3. Đối chiếu tài liệu gốc

| Công trình | Nội dung liên quan đã đối chiếu | Ý nghĩa cho lựa chọn | Giới hạn khi chuyển sang dataset này |
|---|---|---|---|
| Attention-based Deep MIL, ICML 2018 [1] | Attention pooling học ở cấp bag, không cần nhãn từng instance | Có đối chứng tổng hợp vùng rõ ràng | MIL nhị phân không trực tiếp giải quyết lỗi vắng mặt hoặc hình dạng trải dài |
| NTS-Net, ECCV 2018 [2] | Học vùng phân biệt từ nhãn ảnh, kết hợp vùng để phân loại | Local-region learning có cơ sở | Bộ đề xuất vùng và train phức tạp; không tự bảo đảm vùng đúng trên ảnh ống hút |
| WS-DAN, 2019 [3] | Attention crop/drop phục vụ FGVC | Đối chứng khai thác vùng từ nhãn ảnh | Crop/drop có thể đổi ý nghĩa nhãn missing/good; cần adaptation công khai |
| PMG, ECCV 2020 [4] | Học/tổng hợp nhiều mức đặc trưng; progressive train và jigsaw | Có thể khai thác đa mức mà không bắt buộc detector/ROI | Jigsaw phá cấu trúc ống hút nên không bê nguyên làm augmentation giữ nhãn |
| CAL, ICCV 2021 [5] | Học attention qua can thiệp được xây dựng trong mô hình | Không nên xem heatmap đẹp là bằng chứng đủ | Không suy từ che ảnh/attention ra quan hệ nhân quả vật lý |
| GeM, TPAMI 2018 [6] | Pooling tổng quát trung bình/max | Đối chứng pooling gọn | Bài gốc về retrieval; không chứng minh hiệu quả cho phân loại lỗi |
| Anti-aliased CNN, ICML 2019 [7] | Lọc trước downsampling, khảo sát shift stability | Đối chứng chống nhạy vị trí | Lọc có thể làm yếu chi tiết lỗi; cần đo accuracy cùng consistency |
| Hinton et al., KD, 2015 [8] | Soft targets với temperature và supervision nhãn thật | Cơ sở cho chuyển teacher -> student | Student không được bảo đảm giữ mọi kiến thức hoặc chi tiết teacher nhìn thấy |
| Global/Local Logit Distillation, ICCV 2021 [9] | Chưng cất global/local logits và quan hệ không gian | Related work sát, cần đối chiếu trước claim | Global-local + distillation đã có tiền lệ, không tự là tính mới |
| CustomKD, CVPR 2025 [10] | Khoảng cách kiến trúc/năng lực teacher–student có thể cản KD | Teacher lớn hơn chưa chắc cho student tốt hơn | Nghiên cứu LVFM/edge/UDA/SSL; không phải bằng chứng cho teacher CNN của đồ án |
| Gumapac, Sciforum abstract/poster 15-10-2026 [11] | Cùng dataset 2.000 ảnh, fine-tuning 3 CNN và Grad-CAM | Phải vượt mức chỉ so backbone + minh họa heatmap về độ sâu phân tích | Chưa đối chiếu split, không so accuracy trực tiếp; abstract/poster không là benchmark thống nhất |

NTS-Net có thành phần tên Teacher nhưng không được mặc định đồng nhất với
knowledge distillation của [8]. Mỗi công trình có giả định và protocol riêng.
Phần PMG đã đọc cả mô tả train/inference: nhiều bước cập nhật và jigsaw là
thành phần riêng, không chỉ là nối đặc trưng. Không tái hiện thì không gọi bản
ghép tầng đơn giản là kết quả tái lập PMG.

## 4. Đề tài và câu hỏi nghiên cứu sau tổng hợp

Tên lõi: **Nghiên cứu học đặc trưng toàn cục–cục bộ có thông tin vị trí cho
phân loại khuyết tật ống hút bằng CNN.**

Nếu hoàn thành và có kết quả phần KD, tên mở rộng có thể thêm: **và chưng cất
tri thức sang mô hình một lượt suy luận**. Chốt tên cuối theo phần thực sự
hoàn thành, không hứa từ đầu hai đóng góp dương.

- RQ1: CNN toàn ảnh mất lợi ích phân loại nào khi giảm độ phân giải?
- RQ2: Bằng chứng toàn ảnh và local có bổ sung, và cơ chế tổng hợp nào hữu ích?
- RQ3: Thông tin vị trí crop có giúp thêm so với cùng mô hình bỏ vị trí?
- RQ4 (có điều kiện): cùng student một lượt, KD từ global-local có tốt hơn
  CE và KD từ teacher toàn ảnh độ phân giải cao không, với chi phí thế nào?

Giả thuyết phải được bác bỏ được: local có thể chỉ làm tăng compute, vị trí
có thể học shortcut, KD có thể làm student kém đi. Cả ba là kết quả hợp lệ
nếu protocol và phân tích đủ rõ.

## 5. Kiến trúc được ưu tiên kiểm chứng

### 5.1. Teacher: global-local dùng chung CNN

Ảnh gốc -> một global 224 và bốn crop chồng lấn 224 từ ảnh gốc -> shared
ResNet18 -> g và h_1,...,h_K. Không tạo crop bằng nhãn, không dùng pseudo-mask
phụ thuộc lớp, không gán nhãn ảnh cho từng crop.

Với tọa độ crop chuẩn hóa r_k=(cx,cy,w,h):

$$s_k=v^T\tanh(W_hh_k+W_gg+W_r\phi(r_k)+b),\qquad
a_k=\operatorname{softmax}_k(s),\qquad z_l=\sum_k a_kh_k.$$

$$p_T=\operatorname{softmax}(c([g,z_l])),\qquad L_T=CE(y,p_T).$$

phi là embedding nhỏ. r_k lấy từ thao tác cắt ảnh, không phải bounding box
ống hút hoặc nhãn mới. Tất cả biến đổi hình học áp dụng trước khi tạo crop;
tọa độ mô tả ảnh đã biến đổi. Với lưới cố định, embedding vị trí là prior phụ
thuộc vị trí; có thể học thuộc bố cục, nên bắt buộc đối chứng bỏ tọa độ.

Đây là attention có thông tin vị trí, KHÔNG là chứng minh giữ mọi quan hệ hình
học. Tọa độ vùng thô không thay thế detector hay mô hình đầy đủ của hình dạng
ống hút. Với ảnh missing, quyết định dựa vào toàn bộ biểu diễn ảnh, không
ép tồn tại một patch dương tính. Với bent, nhánh global giữ cấu trúc rộng.

### 5.2. Student: CNN một ảnh toàn cảnh

Khởi đầu student là ResNet18 224, trùng kiến trúc/input với baseline B0.
Mục tiêu chính là giảm số view/chi phí so với teacher, không mặc định giảm
số tham số encoder. MobileNetV3 chỉ là mở rộng riêng nếu đủ thời gian.

Teacher đóng băng, train student bằng:

$$p_T^\tau=\operatorname{softmax}(z_T/\tau),\quad
p_S^\tau=\operatorname{softmax}(z_S/\tau),$$

$$L_S=\alpha CE(y,p_S^1)+(1-\alpha)\tau^2 KL(p_T^\tau\Vert p_S^\tau).$$

Phần soft-loss có gradient theo logit student:

$$\frac{\partial L_{soft}}{\partial z_{S,k}}
=(1-\alpha)\tau(p_{S,k}^\tau-p_{T,k}^\tau).$$

Nhờ đó supervision gồm nhãn thật và thông tin về phân bố dự đoán của teacher.
Đây là KD tiêu chuẩn để bắt đầu, không gọi là loss mới. Không đồng nhất tau
trong KD với temperature calibration của xác suất triển khai.

Teacher và student chỉ học từ Train; chọn checkpoint/alpha/tau bằng Val, không
sinh pseudo-label Test để train. Dùng cùng phiên bản ảnh hình học trước khi
tạo global/crop cho teacher và student. Nếu cache logits, ghi augment/view,
teacher hash và sample ID, không ghép logits từ view/ảnh khác.

Teacher nhìn nhiều pixel hơn; nếu thông tin phân biệt bị mất ở global 224,
KD không có bảo đảm phục hồi nó. Nếu cần thử student 384, phải có baseline
CE 384 tương ứng và báo compute; không thay input chỉ cho student KD.

Phân giải hiệu dụng cũng phải phân biệt với số view: crop rộng 60% ảnh rồi
resize 224 tương ứng mật độ lấy mẫu theo chiều khoảng 224/0,6 = 373 trên
toàn khung, thấp hơn global 512. Vì vậy cấu hình bốn crop này giữ chi tiết
hơn global 224, không hơn global 512 một cách tự động. Nếu muốn crop nhỏ hơn
để tăng chi tiết, phải tính lại độ phủ và compute, rồi thêm đối chứng global
tương ứng. Không quảng bá số view như bằng chứng chất lượng thông tin.

## 6. Thiết kế thí nghiệm để phân biệt đóng góp

### Nhóm A: hiệu quả nhận biết và cơ chế

Giữ B0–B5 của STRAW_RESEARCH_PLAN.md; thêm B6 = B5 + embedding tọa độ crop.
B0/B1 là toàn ảnh 224/512; B2 local mean; B3 global-local mean; B4 attention
local; B5 attention có global. 7 cấu hình × 3 seed = 21 run dự kiến.

Ba đối chiếu cơ chế: B3 với toàn ảnh/local; B5 với B4; B6 với B5.
Đối chiếu ứng dụng chính: family global-local được chọn bằng Val với baseline
toàn ảnh mạnh nhất được chọn bằng Val. Các thay đổi family phải khóa trước Test.

Bổ sung ít nhất một đối chứng đã có tiền lệ trong cùng protocol: B4 dùng
attention MIL dạng chuẩn, ghi rõ adaptation đa lớp và nhánh global, không
tuyên bố tái lập nguyên bài MIL nhị phân. Nếu muốn claim vượt WS-DAN/PMG/NTS,
phải chạy bản tương ứng trên cùng split hoặc thu hẹp claim; không dùng điểm
từ dataset/split khác như bảng cạnh tranh trực tiếp.

### Nhóm B: chuyển tri thức (chỉ mở sau khi teacher hữu ích trên Val)

| Cấu hình | Student | Teacher | Mục đích |
|---|---|---|---|
| S0 = B0 | ResNet18 224, CE | Không | Student đối chứng, dùng lại đúng run B0 |
| S1 | Cùng student, KD | ResNet18 global 512 B1 | Lợi ích KD từ teacher toàn ảnh |
| S2 | Cùng student, KD | Global-local chọn bằng Val | Lợi ích của nguồn tri thức ưu tiên |

S1/S2 × 3 seed = 6 run thêm; teacher đã được train ở nhóm A. Tổng mục tiêu
21+6=27 run xác nhận, chưa gồm tuning/pilot. Không khẳng định 27 run chắc
chạy xong trong 4 tuần; profile Kaggle trước. Teacher mỗi seed cùng family
đã chọn, ghi chất lượng teacher; chênh S2–S1 có thể phụ thuộc chất lượng
teacher, không tự chứng minh cơ chế global-local là nguyên nhân duy nhất.

Khóa một ngân sách alpha/tau nhỏ và bằng nhau cho S1/S2, ví dụ tối đa ba cặp
đã khai báo. Tính cả chi phí train teacher, cache và student trong báo cáo.
Không cho KD nhiều lượt tìm cấu hình hơn CE rồi quy mọi lợi ích cho loss.

### Nhóm C: thách thức baseline và tính ổn định

Đầu tiên pilot thêm EfficientNet-B0 pretrained làm baseline thực dụng. Nếu
ngang hoặc hơn ResNet18, báo đầy đủ và không giấu để giữ ưu thế phương pháp.
Nếu muốn kết luận không phụ thuộc backbone, phải lặp baseline và phương pháp
trên backbone thứ hai với đủ seed; một pilot không đủ cho claim đó.

GeM hoặc anti-aliasing là đối chứng gọn nếu pilot chỉ ra pooling/shift sensitivity.
Không chạy tích Descartes backbone × pooling × downsampling × KD × crop.
Mở rộng robustness cố định dùng ảnh vẫn giữ đủ sản phẩm; accuracy/recall gốc
vẫn là chỉ số cần báo, không tối ưu riêng consistency bằng nghiệm đoán hằng.

## 7. Đóng góp được phép phát biểu sau thực nghiệm

1. **Biểu diễn:** xác định bằng chứng về vai trò của toàn cảnh/chi tiết/vị trí
   cho từng loại lỗi, kèm thiết kế CNN được kiểm chứng bằng ablation.
2. **Hiệu năng–chi phí:** nếu KD thành công, student một lượt cải thiện so
   với CE cùng kiến trúc/input, và đánh đổi với teacher được đo thực tế.
3. **Thực nghiệm:** protocol nhóm ảnh, lặp seed, uncertainty, đối chứng mạnh,
   phân tích ảnh sai và công bố cấu hình/prediction tái lập.

Chưa được phát biểu: thuật toán hoàn toàn mới, SOTA, tăng lớn có bảo đảm,
attention định vị đúng, tổng quát mọi sản phẩm hoặc đáp ứng dây chuyền nhà máy.
Global-local, vị trí và KD đều có tiền lệ; novelty cần đối chiếu chính xác
với [2–5,9], không suy ra từ việc áp dụng lên dataset này lần đầu.

Nếu B6 không hơn B5, loại phần vị trí khỏi phương pháp cuối. Nếu KD không
giúp, báo giới hạn và không gắn KD thành đóng góp dương trong tên/kết luận.
Nếu baseline đạt trần quan sát, ưu tiên phân tích lỗi còn lại/độ tin cậy/chi
phí, không bịa mục tiêu tăng điểm lớn.

## 8. Kết quả cần trình bày và mốc dừng

- Bảng chính: macro-F1, accuracy, recall 5 lớp, FNR/FPR suy từ quyết định 5
  lớp, số lỗi thực tế; mean±SD seed và paired group-bootstrap CI.
- Bảng cơ chế: B3/B4/B5/B6, cùng view và cùng backbone.
- Bảng KD: S0/S1/S2, chất lượng teacher, latency/VRAM/params, tổng train cost.
- Đồ thị hiệu năng–latency; learning curves; confusion matrix; ảnh thành
  công/thất bại. Attention là chẩn đoán, không tính định vị khi thiếu GT.
- Một test nội bộ ~299 ảnh còn hạn chế độ phân giải thống kê; không coi
  vài mẫu khác nhau hoặc ba seed như bằng chứng cải thiện lớn/chắc chắn.

Mốc 1: audit/related work/baseline/profiling. Mốc 2: pilot global-local.
Mốc 3: lặp nhóm A sau khi khóa recipe. Mốc 4: KD nếu qua điều kiện và còn
compute, song song viết luận văn. Nếu thời gian ngắn, hoàn thành nhóm A và
phân tích sâu trước khi thêm nhóm B. Test không dùng làm điều kiện mở KD.

Hiện chưa có mô hình B2–B6, trainer tương ứng, teacher/student hoặc kết quả
mới. Tài liệu này tổng hợp nghiên cứu và cập nhật đề cương; không thay đổi
dataset hoặc khởi chạy training.

## Tài liệu nguồn

[1] https://proceedings.mlr.press/v80/ilse18a.html

[2] https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.php

[3] https://arxiv.org/abs/1901.09891

[4] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf

[5] https://arxiv.org/abs/2108.08728

[6] https://arxiv.org/abs/1711.02512

[7] https://proceedings.mlr.press/v97/zhang19a.html

[8] https://arxiv.org/abs/1503.02531

[9] https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Distilling_Global_and_Local_Logits_With_Densely_Connected_Relations_ICCV_2021_paper.pdf

[10] https://arxiv.org/abs/2503.18244

[11] https://sciforum.net/paper/34988

## Bổ sung thư mục tài liệu cần đọc toàn văn

Đã xác định thêm công trình của Gutierrez, Alegre và Macawile:
*CNN Based Classification of Straw Defects in Beverage Box Packaging*, ISCMI
2025. Tên/tác giả được xác nhận trong chương trình hội nghị chính thức:
https://www.iscmi.us/prog.pdf (mã RB142); DOI được ghi nhận là
https://doi.org/10.1109/ISCMI67495.2025.11358586 . Chưa truy cập được toàn văn
từ nhà xuất bản ở lần rà này; không dùng số kết quả từ bản tóm tắt thứ cấp
để so trực tiếp hoặc coi protocol đã được đối chiếu. Đây là việc cần hoàn tất
trong related work, không tự đánh dấu review hoàn thành.
