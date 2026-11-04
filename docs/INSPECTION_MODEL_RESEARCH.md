# Toán học trong mô hình và quá trình huấn luyện kiểm tra lỗi sản phẩm

> **Tài liệu hướng trước:** kế hoạch chính hiện ở
> [STRAW_RESEARCH_PLAN.md](STRAW_RESEARCH_PLAN.md). Mục tiêu đã điều chỉnh về
> phân loại trạng thái lỗi; MixStyle/chuyển miền thương hiệu dưới đây là lịch
> sử phương án, không còn là hướng bắt buộc của khóa luận.

Ngày định hướng: 2026-10-18. Tài liệu này ưu tiên **biểu diễn, biến đổi đặc
trưng và hàm loss** theo yêu cầu người dùng; phần kiểm soát ngưỡng sau dự đoán
là phụ trợ. Không gọi các prototype hoặc kiểm thử đại số là kết quả mô hình tốt.

**Phạm vi đã chốt:** chỉ sử dụng bộ Juice Box Straw Defects hiện có (2.000
ảnh, 8 thương hiệu, 5 lớp). Không yêu cầu thêm bộ dữ liệu sản phẩm khác. Khả
năng tổng quát được nghiên cứu giữa thương hiệu/điều kiện ảnh trong phạm vi
dataset này; chưa có bằng chứng chuyển sang sản phẩm khác hoặc nhà máy thực tế.

## 1. Đối tượng nghiên cứu và giả thuyết có thể bác bỏ

Đề tài gợi ý: **Học biểu diễn ổn định trước biến thiên thống kê đặc trưng cho
phân loại khuyết tật ống hút và đánh giá khả năng tổng quát giữa thương hiệu**.

Với mỗi bài toán sản phẩm, có ảnh x, nhãn y thuộc {0,...,C-1}, và domain d nếu
quan sát được (thương hiệu/camera/lô/điều kiện chụp). Cấu trúc nhãn được khai
báo riêng cho từng dataset. Không giả định các loại lỗi của hai sản phẩm có
cùng ngữ nghĩa hoặc một mạng đã train cho ống hút nhận ra lỗi kim loại ngay.

Giả thuyết H1: một phần thay đổi giữa nguồn ảnh thể hiện ở thống kê kênh của
đặc trưng tầng đầu. Biến đổi những thống kê đó khi train có thể làm mô hình bớt
phụ thuộc chúng và cải thiện đánh giá ở điều kiện giữ riêng.

Giả thuyết H2: chọn ảnh cho thống kê trộn trong cùng lớp giúp hạn chế thay đổi
phần thông tin có liên quan đến loại lỗi so với trộn ngẫu nhiên khác lớp.

Giả thuyết H3: ràng buộc nhất quán giữa dự đoán gốc và dự đoán sau trộn giảm độ
nhạy không cần thiết, đồng thời supervision giữ khả năng phân biệt lớp.

Đây là giả thuyết ứng dụng. **Mean/std cũng có thể chứa dấu hiệu lỗi**; nếu
chúng là tín hiệu cần thiết, biến đổi có thể làm kết quả kém đi. Cả H1/H2/H3
đều cần kiểm chứng bằng ablation; không được mặc định thành kết luận.

## 2. Thay đổi ngay trong model: thống kê đặc trưng

Tại một tầng CNN, lấy F_i thuộc R^(C×H×W). Theo từng kênh c:

$$\mu_{ic}=\frac{1}{HW}\sum_{h,w}F_{ichw},\qquad
\sigma_{ic}=\sqrt{\frac{1}{HW}\sum_{h,w}(F_{ichw}-\mu_{ic})^2+\epsilon}.$$

Phần không gian chuẩn hóa:

$$Z_i=(F_i-\mu_i)/\sigma_i.$$

Chọn donor j và lambda ~ Beta(a,a). Dùng độ mạnh kappa thuộc [0,1]:

$$w_i=\kappa(1-\lambda_i),\quad
\widetilde\mu_i=(1-w_i)\mu_i+w_i\mu_j,\quad
\widetilde\sigma_i=(1-w_i)\sigma_i+w_i\sigma_j,$$

$$\widetilde F_i=\widetilde\sigma_i Z_i+\widetilde\mu_i.$$

Đây là cơ chế kiểu MixStyle, có nguồn gốc rõ ràng từ Zhou et al. (ICLR 2021),
không phải phát minh của dự án. Biến thể điều kiện lớp và loss bên dưới cần
được rà related work trước khi có bất kỳ tuyên bố về tính mới.

**Mệnh đề đại số:** với sigma_tilde > 0,

$$\frac{\widetilde F_i-\widetilde\mu_i}{\widetilde\sigma_i}=Z_i.$$

Chứng minh: thay trực tiếp công thức F_tilde và rút gọn. Vì vậy phép biến đổi
giữ nguyên bản đồ không gian đã chuẩn hóa của recipient theo thống kê được
dùng. Điều này **không chứng minh** nó bảo toàn mọi thông tin ngữ nghĩa của
lỗi. Chỉ dưới mô hình giả thiết F=scale(style)×content+shift(style) mới có thể
diễn giải sự thay đổi này là thay style trong khi giữ content.

Kappa=0 là identity; kappa=1 là phép trộn đầy đủ. Kappa nhỏ giúp khảo sát độ
nhạy cục bộ. Cặp (mu_tilde,sigma_tilde) nằm trên đoạn nối hai cặp thống kê;
không suy ra ảnh/đặc trưng mới tương ứng một camera vật lý có thực.

Prototype chèn sau `layer1` và `layer2` của ResNet18, không thay classifier.
Module không thêm tham số học. Tắt hoàn toàn khi eval: một lượt ResNet bình
thường, không cần ảnh donor, nhãn hoặc domain của ảnh mới.

## 3. Chọn donor và quy trình train

Ba chính sách được triển khai riêng:

- random: donor theo hoán vị batch, làm baseline kiểu MixStyle;
- same_class: j khác i và y_j=y_i;
- cross_domain_same_class: thêm d_j khác d_i, chỉ khi có domain thật.

Nếu không có donor hợp lệ, giữ nguyên mẫu và báo tỷ lệ cặp có hiệu lực khi
train đầy đủ. Không bịa domain, không tự ghép khác lớp rồi gọi là cùng lớp.
Same-class dùng nhãn **chỉ khi train**, như supervision thông thường; inference
tắt augmentation nên không cần biết nhãn thật. Một mẫu duy nhất mỗi lớp trong
batch làm biến thể điều kiện trở thành identity: batch sampler phải được thiết
kế có ít nhất hai đơn vị khác nhau mỗi lớp được chọn. Các biến thể phải dùng
cùng sampler để tránh nhầm lợi ích lấy mẫu với lợi ích thuật toán.

Trong code, mu và sigma được `detach` trước khi tái tổ hợp. Vì vậy gradient
không đi từ recipient qua thống kê donor; đây là quy ước stop-gradient rõ
ràng. Không được mô tả gradient đó là đạo hàm đầy đủ của công thức không detach.

## 4. Thay đổi hàm loss và giải thích tại sao

Cho p_i=softmax(f_theta(x_i)), p_tilde_i là dự đoán khi biến đổi đặc trưng:

$$L_{sup}=\frac12 CE(y_i,p_i)+\frac12 CE(y_i,\widetilde p_i),$$

$$m_i=\frac{p_i+\widetilde p_i}{2},\qquad
JS(p_i,\widetilde p_i)=\frac12 KL(p_i\Vert m_i)+\frac12 KL(\widetilde p_i\Vert m_i),$$

$$L=L_{sup}+\beta\,JS(p_i,\widetilde p_i).$$

- CE của cả hai nhánh buộc dự đoán đúng nhãn sau biến đổi, không chỉ làm hai
  nhánh đồng ý với nhau.
- JS đối xứng, không âm và bị chặn trên bởi log(2); được tính ổn định từ logits.
- Chỉ dùng JS sẽ cho phép nghiệm tầm thường: mọi ảnh có cùng phân phối đầu
  ra. CE làm nghiệm đó bị phạt khi dataset có nhiều lớp. Điều này không bảo
  đảm SGD luôn tìm được nghiệm tối ưu hoặc loại bỏ mọi dạng collapse.

### Khai triển cục bộ: consistency phạt điều gì?

Với p_k>0 và nhiễu xác suất nhỏ u, tổng u_k=0:

$$JS(p,p+u)=\frac18\sum_k\frac{u_k^2}{p_k}+O(\lVert u\rVert^3).$$

Giả sử vector thống kê style s thay đổi một lượng nhỏ delta và mô hình trơn:

$$u\approx J_s p\,\delta,\qquad
JS\approx\frac18\delta^T(J_s p)^T\operatorname{diag}(1/p)(J_s p)\delta.$$

Đây là giải thích toán học: ở lân cận nhỏ, consistency phạt độ nhạy của dự
đoán theo các hướng style được biến đổi. Nó không phạt mọi hướng nhiễu ảnh và
không phải bảo đảm Lipschitz toàn cục. Với trộn mạnh hoặc p rất sát 0, khai
triển bậc hai không đủ để kết luận; phải dùng loss/thí nghiệm thực tế.

## 5. Thiết kế thí nghiệm gắn đúng vào từng giả thuyết

Giữ một backbone ban đầu, cùng pretrained weights, input resolution, sampler,
learning rate, scheduler, số optimizer step và checkpoint-selection rule.

| Phương án | Trộn thống kê | Điều kiện lớp | JS | Câu hỏi |
|---|---|---|---|---|
| A | Không | Không | Không | ERM hai nhánh đối chứng |
| B | Có | Không | Không | Trộn thống kê đơn thuần có ích không? |
| C | Có | Có | Không | Ghép cùng lớp có thêm lợi ích không? |
| D | Có | Không | Có | JS có ích với trộn thường không? |
| E | Có | Có | Có | Kết hợp có lợi và tương tác gì không? |

Tất cả có hai forward khi train để kiểm soát số lượt xem ảnh/độ sâu graph;
ERM dùng hai nhánh không trộn. Báo cả GPU time và memory, không gọi hai lượt
forward bằng một lượt về chi phí. Dùng RNG riêng cho DataLoader để khác biệt
số lần rút ngẫu nhiên của augmentation không làm thay đổi thứ tự ảnh.

Chạy pilot một seed trước; mở 3 seed sau khi xác nhận dữ liệu, batch và chi
phí. Kiểm tra strength {0,0.25,0.5,1}, beta {0,0.1,0.5,1} theo một ngân sách
tìm kiếm Validation hữu hạn, không chọn theo Test. Không cần chạy tích Descartes
mọi tham số/backbone nếu không trả lời câu hỏi nghiên cứu.

Chỉ số chính: macro F1/recall từng lớp ở bài đa lớp; AUROC, PR-AUC và FNR tại
operating point khóa từ Validation ở bài nhị phân. Báo sai khác giữa source
Validation và thương hiệu giữ riêng; paired group bootstrap; mean±SD qua seed.
Corruption tổng hợp là chẩn đoán độ nhạy, không thay thế chuyển miền thật.

Để khảo sát cơ chế, đo biến thiên dự đoán theo thay đổi thống kê và phân tích
nhầm lớp trước/sau; accuracy của một bộ phân loại domain trên frozen features
chỉ là chẩn đoán. Không thể từ việc domain khó dự đoán suy ra biểu diễn là nhân quả.

## 6. Tổng quát trong phạm vi một bộ dữ liệu

Chỉ sử dụng bộ ống hút hiện có. Có ba mức bằng chứng cần phân biệt:

1. **Cùng phân phối:** split theo nhóm ảnh, stratify theo thương hiệu/lớp,
   kiểm tra các thương hiệu đã xuất hiện trong Train. Đây là baseline nội bộ.
2. **Giữ riêng thương hiệu (LOBO):** mỗi fold train trên 7 thương hiệu và đánh
   giá trên thương hiệu thứ 8. Checkpoint, hyperparameter và temperature nếu
   dùng phải được chọn chỉ từ dữ liệu phát triển của 7 thương hiệu nguồn.
   Không chọn cấu hình bằng kết quả thương hiệu đích rồi vẫn gọi đó là chưa thấy.
3. **Ảnh suy giảm tổng hợp:** kiểm tra nhạy cảm với mờ/tối/nhiễu/nén tại tham
   số đã cố định. Đây là stress test, không chứng minh tổng quát với camera mới.

Đơn vị độc lập và nhóm ảnh liên quan cần được kiểm tra trước khi chia tập.
Giữ riêng thương hiệu không chứng minh đã giữ riêng camera, phiên chụp hay
lô sản xuất nếu không có metadata tương ứng. Dataset đã được dùng trong các
phiên phát triển trước; kết quả mới phải được mô tả là benchmark nội bộ.

Mã và phép biến đổi có thể được thiết kế không gắn cứng số lớp để dùng lại
về sau. Tính dùng lại của mã khác với bằng chứng hiệu năng: trong khóa luận
này chỉ kết luận hiệu quả cho phân loại lỗi ống hút và chuyển miền thương
hiệu đã thực sự được kiểm tra, không kết luận hiệu quả trên mọi sản phẩm.

## 7. Trạng thái triển khai

- Đã có `src/inspection_research/feature_style.py`: module trộn, pairing,
  StyleResNet18, loss hai nhánh và JS.
- 14 kiểm thử đã PASS: đẳng thức chuẩn hóa, stop-gradient, giới hạn JS, khai
  triển cục bộ, backward của ResNet và inference không nhãn cho các số lớp khác nhau.
- `scripts/research/feature_style_step.py` kiểm tra một bước train trên ảnh
  thật; random initialization, không dùng để kết luận hiệu năng.
- Chưa có trainer/runner nhiều epoch và notebook Kaggle cho phương pháp mới;
  pipeline straw 72 run cũ chưa tự sử dụng module này.
- Không bổ sung KolektorSDD2 hoặc dataset sản phẩm khác vào phạm vi hiện tại;
  chưa có bằng chứng cải thiện chuyển miền thương hiệu từ prototype mới.
- Các module probability/decision/risk_control trong cùng package là công cụ
  phụ trợ, không phải đóng góp model/training chính của định hướng này.

Thay đổi có giá trị chỉ khi chuỗi **vấn đề → giả thiết → công thức → code →
ablation → kết quả → giới hạn áp dụng** được kiểm chứng đầy đủ.

## Tài liệu gốc

- MixStyle, paper/code của tác giả: https://github.com/KaiyangZhou/mixstyle-release
  và https://arxiv.org/abs/2104.02008
- DomainBed và vấn đề so sánh/model selection công bằng:
  https://openreview.net/pdf?id=lQdXeXDoWtI
