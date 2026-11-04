# Khóa luận tốt nghiệp nghiên cứu phân loại khuyết tật ống hút

## Thông tin báo cáo

**Tên đề tài dự kiến:** Nghiên cứu kết hợp đặc trưng toàn cục và cục bộ trong mạng CNN cho phân loại khuyết tật ống hút trên bao bì hộp đồ uống  
**Phạm vi:** Bộ dữ liệu ảnh hiện có với năm nhãn gốc; thực nghiệm CNN B0–B6; nguyên mẫu mô phỏng băng chuyền và suy luận TensorRT.  
**Trạng thái:** Báo cáo tổng quan kỹ thuật và nghiên cứu đang thực hiện; chưa phải bản luận văn đã hoàn tất.

## Tóm tắt

Đề tài xây dựng và đánh giá hệ thống phân loại ảnh sản phẩm thành năm trạng thái: `good`, `bent`, `loose`, `missing`, `pierced`. Trọng tâm nghiên cứu hiện tại là xác định liệu việc giữ độ phân giải ảnh, kết hợp biểu diễn toàn ảnh với các vùng ảnh phủ cố định, và dùng attention có điều kiện ngữ cảnh hoặc vị trí có cải thiện khả năng phân loại so với ResNet18 toàn ảnh hay không. Bảy cấu hình B0–B6 được thiết kế sao cho mỗi bước thêm thành phần có một đối chứng gần tương ứng. Thương hiệu chỉ là metadata mô tả; không phải đầu vào hoặc nhãn dự đoán.

Bộ dữ liệu nội bộ có 2.000 ảnh, tám thương hiệu và năm lớp cân bằng. Thí nghiệm xác nhận `confirmatory_kaggle_v1` dùng 1.401 ảnh train, 300 validation và 299 test; toàn bộ 21 run (B0–B6 × ba seed 42/43/44) đã hoàn tất, được freeze và có test output. Trên test sạch, B1 512 đứng đầu với macro-F1 99,34% ± 0,00 và accuracy 99,33% ± 0,00; B6 đạt macro-F1 97,99% ± 0,89. So sánh paired đã đăng ký giữa B6 và baseline B1 cho chênh lệch -1,35 điểm phần trăm, CI 95% [-2,63; -0,32], tức bằng chứng trên protocol này không ủng hộ B6 cải thiện so với B1. Ba seed dùng cùng 299 ảnh test, không phải 897 ảnh độc lập. Lô 30 ảnh Bent bạn chạy đạt 60%, nhưng kiểm toán sau đó phát hiện ảnh degraded bị xoay 90° do bỏ qua EXIF; đồng thời app băng chuyền mặc định dùng artifact B1 384 khác checkpoint B1 512 trong bảng test. Vì vậy 60% không phải phép đo hợp lệ để kết luận hiệu năng của B1 512 trên ảnh suy giảm đúng chiều.

Nhánh triển khai cũ còn có TensorRT/DeepStream, RTSP phát lại ảnh tĩnh và demo băng chuyền canvas. Nhánh TensorRT đạt macro-F1 99,016% trên validation replay; RTSP quan sát 31,03 FPS. Trên 30 ảnh suy giảm có nhãn `bent` qua demo, chỉ 18 ảnh đúng lớp (60,0%), 10 ảnh nhầm loại lỗi và 2 ảnh bị kết luận là tốt. Đây là chẩn đoán một lô, không phải test độc lập. Bộ suy giảm cũ bị ghi đè ảnh giữa thương hiệu do tên file trùng, nên kết quả Factory Degraded cũ bị loại. Bộ v2 đã tạo lại đủ 300 ảnh duy nhất cho validation và test, hiệu chuẩn và đánh giá lại. Trên test suy giảm v2, full5 có FNR 4,58%, còn 2-stage cascade giữ FNR 0% nhưng chuyển 97% mẫu sang review; ngưỡng cascade hiện tại vì thế chưa hữu dụng cho vận hành. Test sạch cũng cho cascade review 83%. Không gộp số liệu của các nhánh/checkpoint khác nhau vào kết quả xác nhận B0–B6.

**Kết luận hiện thời:** dự án đã có nền tảng nghiên cứu và nguyên mẫu kỹ thuật hoạt động; chưa đủ bằng chứng để kết luận kiến trúc đề xuất cải thiện mô hình, đạt yêu cầu chính thức của trường, hoặc sẵn sàng cho sản xuất.

## Mục lục

1. Tổng quan và phát biểu bài toán  
2. Cơ sở lý thuyết và phương pháp đề xuất  
3. Dữ liệu và quy trình thực nghiệm  
4. Hệ thống phần mềm và nguyên mẫu triển khai  
5. Kết quả hiện có  
6. Đánh giá, giới hạn và rủi ro nghiên cứu  
7. Kế hoạch hoàn thiện và tiêu chí kết luận  
8. Kết luận  
Tài liệu tham khảo và phụ lục

## 1. Tổng quan và phát biểu bài toán

### 1.1 Bối cảnh

Kiểm tra trực quan sản phẩm cần phân biệt sản phẩm đạt yêu cầu với các dạng lỗi có ý nghĩa vận hành. Với bộ dữ liệu này, bài toán được giới hạn vào ảnh hộp đồ uống có ống hút. Hệ thống nhận một ảnh sản phẩm và dự đoán đúng một trong năm lớp. Nhiệm vụ bao gồm hai mức thông tin: sản phẩm tốt hay lỗi, và nếu lỗi thì thuộc lớp nào. Mức tốt/lỗi suy ra từ cùng dự đoán năm lớp: `good` là tốt, bốn lớp còn lại là lỗi.

Đây là phân loại ảnh có giám sát, không phải phát hiện đối tượng, phân đoạn ảnh, phát hiện lỗi chưa biết hoặc kiểm tra mọi loại sản phẩm. Dataset không cung cấp nhãn hộp giới hạn lỗi để tính mAP/IoU. Attention map hoặc crop có thể được phân tích như bằng chứng mô hình dùng, nhưng không được gọi là định vị chính xác nếu chưa có ground truth vị trí.

### 1.2 Mục tiêu

Mục tiêu tổng quát là thiết kế và đánh giá có kiểm soát các biến thể ResNet18 toàn ảnh và global-local, sau đó đo lợi ích nhận biết cùng chi phí tính toán. Các mục tiêu cụ thể:

- Chuẩn hóa manifest, nhóm ảnh liên quan, tiền xử lý và phép chia train/validation/test có fingerprint.
- So sánh độ phân giải ảnh toàn cục 224 và 512.
- Kiểm tra thông tin local có bổ sung cho biểu diễn global không.
- Tách riêng ảnh hưởng của pooling trung bình, attention, điều kiện ngữ cảnh toàn cục và tọa độ crop.
- Đánh giá hiệu năng từng lớp, bỏ sót lỗi, báo nhầm sản phẩm tốt thành lỗi, độ ổn định theo seed và độ trễ.
- Minh họa đường triển khai bằng worker TensorRT thường trú và cổng kích hoạt ROI trong mô phỏng băng chuyền.

### 1.3 Câu hỏi nghiên cứu

**RQ1.** Tăng độ phân giải toàn ảnh từ 224 lên 512 ảnh hưởng thế nào đến macro-F1 và khả năng nhận từng loại lỗi?  
**RQ2.** Bốn vùng ảnh phủ cố định có mang thông tin bổ sung cho toàn cảnh hay chỉ làm tăng chi phí?  
**RQ3.** Attention dựa vào đặc trưng local, ngữ cảnh global và vị trí crop đóng góp riêng ra sao so với tổng hợp trung bình?  
**RQ4.** Mức thay đổi hiệu năng có đáng chi phí xử lý nhiều view và độ trễ suy luận tăng thêm không?

Các câu hỏi này không mặc định rằng kiến trúc phức tạp hơn sẽ tốt hơn. Kết quả âm hoặc không rõ ràng vẫn có giá trị nếu thiết kế thí nghiệm và diễn giải hợp lệ.

### 1.4 Phạm vi và đóng góp dự kiến

Phạm vi kết luận là hiệu năng phân loại trên bộ dữ liệu hiện tại và protocol nội bộ được mô tả trong báo cáo. Không suy rộng sang mọi thương hiệu, mọi nhà máy hoặc camera khác. Thương hiệu không được dùng làm đặc trưng dự đoán; thông tin này chỉ hỗ trợ mô tả dữ liệu và rà soát phép chia.

Đóng góp có thể bảo vệ sau khi hoàn tất thí nghiệm là: (i) quy trình đối chứng B0–B6 giúp tách tác dụng của độ phân giải và từng thành phần global-local; (ii) đánh giá hiệu năng cùng chi phí dưới protocol nhất quán; (iii) một nguyên mẫu xử lý sự kiện băng chuyền có thể tái lập. Chưa có cơ sở để tuyên bố thuật toán hoàn toàn mới, SOTA, hoặc cải thiện chắc chắn.

### 1.5 Các nhánh khác trong workspace

Workspace còn lưu một dự án phân đoạn khuyết tật bề mặt nhôm trên tập con 3CAD-ANI, cùng pipeline phân loại hai tầng/REVIEW đời trước. Đây là nhánh kỹ thuật và hồ sơ khác, không thuộc bộ dữ liệu, mô hình hoặc kết quả của luận văn ống hút này. Không gộp các chỉ số của chúng vào báo cáo.

## 2. Cơ sở lý thuyết và phương pháp đề xuất

### 2.1 Phân loại ảnh bằng CNN

Với ảnh đầu vào (x), mạng CNN (f_\theta) tạo vector đặc trưng và bộ phân loại tuyến tính sinh logits (z\in\mathbb{R}^{5}). Phân phối dự đoán là:

\[
p(y=c\mid x)=\frac{\exp(z_c)}{\sum_{j=1}^{5}\exp(z_j)},\qquad \hat y=\arg\max_c p(y=c\mid x).
\]

Huấn luyện dùng cross-entropy ở cấp ảnh:

\[
\mathcal{L}_{CE}=-\frac{1}{N}\sum_{i=1}^{N}\log p(y_i\mid x_i).
\]

ResNet18 pretrained trên ImageNet là encoder chung. Batch-normalization giữ running statistics pretrained cố định trong các cấu hình global-local; tham số affine vẫn được học. Đây là lựa chọn kiểm soát biến thiên thống kê khi cùng encoder xử lý nhiều view, không phải khẳng định BN freeze là tối ưu nhất.

### 2.2 Giả thuyết toàn cục và cục bộ

Ảnh toàn cục giữ hình dạng sản phẩm, vị trí tương đối và ngữ cảnh; các vùng local có thể giữ mật độ mẫu cao hơn cho dấu hiệu nhỏ. Nếu đặc trưng cục bộ chỉ chiếm một phần nhỏ bản đồ đặc trưng, phép lấy trung bình không gian có thể làm tín hiệu bị pha loãng. Tuy vậy, crop có thể bỏ mất cấu trúc kéo dài, làm tăng số lần chạy encoder hoặc chứa phần lớn nền. Đây là giả thuyết cần đo, không phải kết quả đã được chứng minh.

Ảnh được chuẩn hóa EXIF và biến đổi nhất quán. Nhánh local tạo bốn crop ở bốn góc; mỗi crop rộng và cao bằng 60% ảnh gốc, lấy từ ảnh gốc trước resize và đưa về 224×224. Các crop phủ cố định không phải vùng ống hút được định vị tự động. Không có ground truth crop-level nên loss chỉ áp lên dự đoán tổng hợp của ảnh.

### 2.3 Attention và tọa độ

Gọi (g=f_\theta(G(x))\) là đặc trưng global, (h_k=f_\theta(P_k(x))\) là đặc trưng crop thứ (k). Tổng hợp local dùng trọng số:

\[
a_k=\frac{\exp(s_k)}{\sum_j\exp(s_j)},\qquad z_l=\sum_k a_k h_k.
\]

Trong B4, điểm (s_k) phụ thuộc đặc trưng crop. Trong B5, điểm còn được điều kiện hóa theo (g), tức ngữ cảnh toàn cảnh có thể thay đổi tầm quan trọng của từng vùng. Trong B6, thêm embedding tọa độ crop chuẩn hóa. Các cơ chế này khảo sát giả thuyết khác nhau; attention không tự đảm bảo chọn đúng ống hút và tọa độ không bảo đảm khôi phục đầy đủ cấu trúc hình học.

### 2.4 Định nghĩa cấu hình B0–B6

| Mã | Đầu vào và tổng hợp | Vai trò đối chứng |
|---|---|---|
| B0 | Ảnh global 224, ResNet18 | Baseline gọn |
| B1 | Ảnh global 512, ResNet18 | Tác dụng độ phân giải |
| B2 | Bốn crop local 224, mean pooling | Local không có global |
| B3 | Global 224 + bốn local, mean pooling, ghép vector | Bổ sung ngữ cảnh và local |
| B4 | Như B3, trọng số local từ đặc trưng local | Attention không điều kiện global |
| B5 | Như B4, thêm global vào điểm attention | Tác dụng điều kiện ngữ cảnh |
| B6 | Như B5, thêm tọa độ crop | Tác dụng thông tin vị trí |

Mọi biến thể dùng cùng họ encoder ResNet18 và cùng nhãn đầu ra. Nhiều view làm tăng chi phí xử lý dù encoder được chia sẻ. So sánh cần báo số tham số, latency và peak memory cùng macro-F1.

### 2.5 Chỉ số đánh giá

Macro-F1 là trung bình F1 của năm lớp, để mỗi lớp có trọng số ngang nhau. Accuracy là tỷ lệ dự đoán đúng trên tổng số ảnh. Báo precision/recall/F1 theo từng lớp và ma trận nhầm lẫn để chỉ ra loại lỗi bị nhầm.

Khi gộp bốn lớp lỗi thành Defect:

\[
FNR=\frac{FN}{N_{defect}},\qquad FPR=\frac{FP}{N_{good}}.
\]

FN là ảnh lỗi bị dự đoán `good`; FP là ảnh `good` bị dự đoán thành một trong bốn lớp lỗi. Nhầm `bent` thành `loose` là lỗi phân loại loại lỗi, không phải false negative của kiểm tra tốt/lỗi; cần báo riêng. Mean ± độ lệch chuẩn qua seed mô tả dao động do huấn luyện, còn bootstrap theo nhóm ảnh mô tả bất định mẫu; hai đại lượng không thay thế nhau.

## 3. Dữ liệu và quy trình thực nghiệm

### 3.1 Bộ dữ liệu

Bản mô tả hiện hành ghi nhận 2.000 ảnh, tám thương hiệu và năm lớp, mỗi tổ hợp thương hiệu × lớp có 50 ảnh. Tổng thể cân bằng theo lớp: 400 ảnh cho mỗi nhãn. Nhãn là `good`, `bent`, `loose`, `missing`, `pierced`; thứ tự đầu ra của mô hình là 0=`good`, 1=`bent`, 2=`loose`, 3=`missing`, 4=`pierced`. Thư mục ảnh gốc và manifest trong workspace đều có nhãn `Pierced`. Bản demo trước đây dịch lớp này thành “Ống hút đâm xuyên”, dễ bị hiểu nhầm là một nhãn tiếng Việt khác; giao diện hiện giữ nguyên tên nhãn gốc `Pierced` cho đến khi có bảng thuật ngữ được xác nhận. Cần dùng định nghĩa từ nguồn gốc dữ liệu và ví dụ khó để chuẩn hóa ranh giới giữa `loose` và `pierced`; báo cáo không tự diễn giải `pierced` thành lỗ thủng thân ống hút.

Nguồn gốc bộ ảnh, điều kiện sử dụng và quyền tái phân phối chưa được ghi nhận đầy đủ trong review ledger. Audit có 76 cặp gần nhau theo dHash; nhóm liên quan hiện là heuristic, chưa được xác nhận thủ công hoặc theo phiên chụp. Vì vậy phép chia giảm nguy cơ trùng nội dung đã nhận diện nhưng không chứng minh độc lập tuyệt đối giữa các ảnh.

### 3.2 Phép chia và kiểm soát rò rỉ

Protocol confirmatory dùng 1.401 ảnh train, 300 validation và 299 test (tổng 2.000). Artifact ghi protocol fingerprint `b64d8b501809e386` và experiment fingerprint `c4a9165375a2de8f`. Manifest chứa sample ID, nhãn, group ID và SHA; code kiểm tra sample ID/hàm băm trùng và không cho group ID đi qua nhiều tập. Đây là protocol của bảng B0–B6 bên dưới; không trộn với protocol cũ 1.400/300/300 hoặc các checkpoint triển khai đời trước.

Dataset đã có lịch sử được dùng trong phát triển trước đó. Vì vậy test hiện tại là **test nội bộ theo protocol đã khóa**, không phải test ngoài độc lập chưa từng nhìn thấy. Chọn checkpoint và cấu hình bằng validation; chỉ sau khi hoàn tất rà soát và freeze mới mở test.

| Tập | Số ảnh | Tỷ lệ xấp xỉ | Vai trò |
|---|---:|---:|---|
| Train | 1.401 | 70,05% | Cập nhật trọng số |
| Validation | 300 | 15,0% | Chọn checkpoint/cấu hình và phân tích phát triển |
| Test | 299 | 14,95% | Đánh giá sau khi freeze |

### 3.3 Huấn luyện

Cấu hình xác nhận gồm 7 mô hình × 3 seed (42, 43, 44) = 21 run, tất cả đã hoàn tất. Số epoch tối đa 40, early stopping patience 10. Dùng AdamW, learning rate encoder (10^{-4}), head (10^{-3}), weight decay (10^{-4}), pretrained ResNet18, mixed precision trên CUDA và augmentation nhất quán giữa view. Batch hiệu dụng là 32 ảnh; B1 dùng batch 2 và tích lũy gradient 16 bước, các cấu hình khác batch 4 và tích lũy 8 bước. Đây là protocol đã chạy, không phải khẳng định tối ưu toàn cục.

Lịch sử học B1 seed 42 có 25 epoch: train cross-entropy giảm từ 1,196 xuống gần 0; validation cross-entropy thấp nhất 0,0128 tại epoch 8, sau đó dao động; validation macro-F1 đạt 0,99682 từ epoch 15. Các đường cong cho thấy mô hình khớp rất nhanh trên bộ dữ liệu hiện tại và train CE tiếp tục tiến gần 0 trong khi validation CE không giảm đơn điệu. Đây là dấu hiệu cần thận trọng về overfit/độ dễ của tập, nhưng riêng đường cong không chứng minh rò rỉ ảnh. Checkpoint của mỗi run được chọn theo validation macro-F1; biểu đồ dưới đây chỉ minh họa một run đại diện, không thay cho trung bình ba seed.

![Hình 3.2. Train CE, validation CE và validation macro-F1 theo epoch của B1 seed 42](docs/assets/straw_overview/b1_s42_learning_curves.png)

Pilot phát triển có recipe ngắn riêng: seed 42, ba epoch, patience 3, chỉ Train/Validation. Pilot không chạy Test. Checkpoint chọn bằng macro-F1 validation; validation cross-entropy là tiêu chí phụ khi hòa. Resume lưu trạng thái huấn luyện; fingerprint source/config/protocol/cache ngăn dùng lẫn output sau khi thay đổi.

### 3.4 Phân tích thống kê và benchmark dự kiến

Kết quả được tổng hợp mean ± SD qua ba seed; mỗi seed đánh giá cùng 299 ảnh test. So sánh paired B6 với B1 dùng bootstrap theo nhóm ảnh (1.000 lần), điều kiện theo các seed đã huấn luyện; không coi ba seed là ba tập test độc lập. Benchmark batch 1 đã đo decode ảnh, tạo view, chuẩn hóa, transfer và forward trên Tesla T4, với warmup và lặp lại. Các số liệu và giới hạn được trình bày ở Mục 5.

## 4. Hệ thống phần mềm và nguyên mẫu triển khai

### 4.1 Kiến trúc phần mềm

Mã nghiên cứu được chia thành data/protocol, model, runner, đánh giá và đóng gói Kaggle. Notebook/gói Kaggle đọc dataset được gắn làm Input, không nhúng ảnh. Checkpoint và optimizer có thể tiếp tục qua phiên nếu output archive được lưu rồi gắn vào phiên sau; Kaggle không tự đồng bộ trạng thái chưa lưu.

Các kiểm thử xác minh tạo crop, đổi vị trí crop sau flip, gradient encoder/attention/tọa độ, resume, provenance và chặn test trước freeze. Hồ sơ verification ghi 22 kiểm thử global-local và 15 kiểm thử gói Kaggle đạt. Smoke toàn luồng bảy cấu hình cũng chạy trên tập con nhỏ; smoke không phải kết quả khoa học.

### 4.2 Nhánh TensorRT và DeepStream

Nhánh triển khai trước dùng checkpoint ResNet18 khác với mô hình B0–B6 hiện tại. Checkpoint được chuyển ONNX rồi compile engine TensorRT FP16 và chạy DeepStream. Trên 300 ảnh validation phát thành H.264, PyTorch FP16 trên cùng clip có accuracy 99,333%, macro-F1 99,340%, FN=1, FP=0 và nhầm loại lỗi=1. DeepStream có accuracy 99,000%, macro-F1 99,016%, FN=1, FP=0 và nhầm loại lỗi=2. Có một dự đoán khác nhau trên 300 ảnh; decoder và resize/backend cũng khác, nên không quy toàn bộ chênh lệch cho FP16/TensorRT.

RTSP replay xử lý 252 frame, quan sát 31,03 FPS; thời gian từ khởi động ứng dụng đến dự đoán đầu tiên là 1,31 giây. Latency sau decode đến kết quả có p50=2,28 ms và p95=11,31 ms cho toàn bộ frame; sau 20 frame đầu p50=2,28 ms, p95=11,25 ms. Các số liệu không gồm cảm biến, mã hóa camera, truyền mạng, jitter, decode, trình duyệt hoặc PLC. Clip gồm ảnh tĩnh phát nối tiếp, nên không chứng minh tốc độ kiểm tra sản phẩm thật.

### 4.2.1 Đối chiếu runtime trên GPU NVIDIA

Để chọn cấu hình cho máy có GPU NVIDIA, cùng checkpoint B1 web 384 (seed 42, SHA-256 `2a1881fcd9c17201ecfc6b6547b699c07e9d0ff8ba1a250ee943adf0f27cf852`) được chạy trên NVIDIA GeForce RTX 3050 Laptop GPU 4 GB, driver 610.60, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19 và TensorRT 11.4.0.106. Phép đo sử dụng batch 1, 30 warm-up, 150 lượt đo, cùng 300 ảnh validation và cùng byte JPEG đầu vào. Tiền xử lý gồm EXIF transpose, đổi RGB, resize bilinear 384×384, JPEG q90, decode, chuẩn hóa ImageNet. Đầu ra là softmax/argmax; latency không bao gồm HTTP hoặc truyền mạng.

| Runtime | Độ chính xác | Accuracy | Macro-F1 | FN | FP | Sai loại lỗi | p50 tổng | p95 tổng | p99 tổng |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CUDA | FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 7,475 ms | 8,955 ms | 10,481 ms |
| PyTorch CUDA | FP16 | 99,667% | 99,682% | 1 | 0 | 0 | 7,514 ms | 10,306 ms | 10,783 ms |
| ONNX Runtime CUDA EP | FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 8,386 ms | 9,048 ms | 10,008 ms |
| ONNX Runtime CUDA EP | FP16 nội bộ, I/O FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 12,423 ms | 15,710 ms | 25,183 ms |
| TensorRT CUDA | FP16 nội bộ, I/O FP32 | 99,667% | 99,682% | 1 | 0 | 0 | 4,785 ms | 6,709 ms | 7,454 ms |

TensorRT FP16 giảm latency so với PyTorch CUDA FP32 khoảng 36,0% ở p50 và 25,1% ở p95; các chỉ số phân loại trùng nhau trên validation. Cấu hình được chọn cho máy NVIDIA trong phạm vi phép đo là B1 384, ONNX FP16 làm đầu vào build, TensorRT engine FP16, batch 1, I/O FP32. Engine phải được build trên GPU và phiên bản TensorRT đích. PyTorch CUDA FP32 được giữ làm đường dự phòng. ONNX Runtime CUDA đã thực thi trên GPU nhưng không nhanh hơn ở workload batch 1 này.

Đã thử TensorRT INT8 với 200 ảnh Train để hiệu chuẩn, Q/DQ cho convolution và giữ classifier ở FP32. TensorRT 11.4 đọc được graph nhưng builder không tạo serialized engine trên RTX 3050. Vì vậy không có kết quả inference hoặc latency INT8 TensorRT; các số INT8 CPU trong benchmark lịch sử không đại diện cho GPU.

Kết quả cao trên validation/test nội bộ chưa xác nhận tổng quát ngoài dữ liệu. Dataset có 2.000 ảnh thuộc 8 thương hiệu, mỗi thương hiệu có 50 ảnh/lớp; cả 8 thương hiệu xuất hiện trong cả Train, Validation và Test. Audit manifest không thấy trùng SHA, pixel hash hay group ID qua các split; còn 76 cặp dHash gần nhau chưa rà thủ công và group kế thừa là heuristic. Phép đánh giá hiện tại vì thế đo khả năng dự đoán trên thương hiệu đã thấy, không đo khả năng chuyển sang thương hiệu mới. Cần kiểm thử giữ riêng thương hiệu, gom theo phiên chụp/vật thể và đánh giá trên ảnh mới được khóa trước khi đưa ra kết luận triển khai.

### 4.3 Demo băng chuyền và cổng pixel

Demo hiện hành cho phép chọn nhiều ảnh; mỗi ảnh được xếp vào băng chuyền canvas và tự chuyển động, không cần nút Run. Worker TensorRT được nạp một lần và giữ sẵn. Khi chưa có sản phẩm, nhánh CPU phân tích frame kích thước nhỏ; GPU không phân loại mọi frame.

Logic pixel gồm: hiệu chuẩn nền trống; đổi ảnh sang luma; ước lượng thay đổi sáng toàn cảnh; trừ nền và ngưỡng hóa sai khác; morphological closing; tìm thành phần liên thông; lọc theo diện tích/kích thước/fill ratio; ghép thành phần giữa frame theo khoảng cách tâm và giữ track tối đa bốn frame bị mất. Track cần tối thiểu ba quan sát, di chuyển từ phải sang trái, cắt vạch kích hoạt và có ít nhất 90% bbox trong ROI. Mỗi track chỉ phát một sự kiện.

Sau sự kiện ROI, API xác thực bằng chứng gate và deduplicate khóa job/ảnh. Worker TensorRT mặc định của demo nạp `artifacts/b1_stream` với checkpoint SHA-256 `2a1881fcd9c17201ecfc6b6547b699c07e9d0ff8ba1a250ee943adf0f27cf852`, kích thước 384 px; đây là artifact B1 fine-tune triển khai, không trùng hash với B1_s42 512 trong thí nghiệm confirmatory (`f95bbd2b4983f31710db861a61f60b801d8da023ddb1aa4504b31c3bcb52870f`). Engine 512 cùng hash B1_s42 có trong `artifacts/b1_stream_512`, nhưng code preview mặc định chọn `b1_stream`; do đó cần kiểm tra `/health` hoặc manifest khi nói demo đang chạy checkpoint nào. TensorRT nhận ảnh sản phẩm đầy đủ, EXIF-transpose nếu còn metadata, rồi resize vuông theo manifest. Kết quả được gắn lại với track. Nếu track hoặc suy luận lỗi, sản phẩm không được ngầm đánh dấu là tốt.

Gate pixel đã được thử trên cảnh mô phỏng canvas, không phải RTSP camera. CNN phân loại ảnh sản phẩm ban đầu, không phân loại crop/cảnh ghép mà detector tạo ra. Băng chuyền chuyển động trên giao diện chưa tương đương vật thể vật lý đi qua camera.

### 4.4 Smoke test của gate

Kiểm thử logic gồm nền trống/nhiễu, thay đổi sáng toàn cục, chặn gọi trước ROI, đúng một lần mỗi vật thể, sản phẩm kế tiếp và vật thể đứng tại ROI. Browser smoke nạp ba ảnh, ghi nhận ba track và ba sự kiện, mỗi track có 69–70 quan sát và ROI overlap 1,0; cả ba suy luận hoàn tất. Lô 30 ảnh `degraded_Bent` bạn đưa qua demo có 18/30 dự đoán đúng lớp (60,0%), 10/30 nhầm sang lỗi khác và 2/30 bị nhận nhầm là `good`; sáu dự đoán `pierced` là dự đoán sai vì nhãn thật là `bent`. Tuy nhiên, kiểm toán ảnh `degraded_Bent (1).jpg` xác nhận hộp bị xoay ngang. Các ảnh nguồn có EXIF Orientation=6; bộ sinh dùng `Image.open(...).convert("RGB")` mà không `ImageOps.exif_transpose`, rồi lưu JPEG mới làm mất metadata. Vì thế app không thể tự sửa chiều ảnh ở bước upload. Lô 60% này bị nhiễu bởi lỗi tạo đầu vào, không thể dùng để ước lượng độ bền của B1 trên ảnh đúng chiều. Ngoài ra, demo mặc định dùng B1 384, trong khi bảng confirmatory là B1 512. Cần tạo lại lô theo EXIF đúng và xác nhận SHA/model size của engine trước khi chạy lại; hiện chưa có kết quả so sánh mới trên lô đã sửa. Lô này chỉ gồm Bent và không thay thế test độc lập; cũng chưa có video camera có nhãn để tính gate recall, false trigger/phút hoặc tỷ lệ bỏ qua sản phẩm.

| Nhãn thật | Dự đoán Good | Bent | Loose | Missing | Pierced | Đúng lớp |
|---|---:|---:|---:|---:|---:|---:|
| Bent, 30 ảnh suy giảm | 2 | 18 | 3 | 1 | 6 | 18/30 (60,0%) |

**Kiểm toán bộ ảnh suy giảm và tiền xử lý:** bộ test degraded cũ có 300 dòng nhưng chỉ 185 ảnh duy nhất; validation cũ có 182 ảnh duy nhất vì tên sinh chỉ dùng tên gốc và tên này lặp giữa thương hiệu. Bộ tạo đã được sửa để thêm thương hiệu và mã băm; `val_degraded_v2` và `test_degraded_v2` hiện có 300/300 đường dẫn duy nhất, cân bằng 60 ảnh mỗi lớp. Ngưỡng được hiệu chuẩn lại trên v2. Ngoài ra, 300 ảnh test nguồn đều có EXIF Orientation=6. Các checkpoint ResNet18 cũ được huấn luyện trước khi thêm EXIF transpose, nên khi đánh giá chúng phải đọc ảnh theo đúng preprocessing lúc train (không transpose); nhánh B1 global-local mới dùng EXIF transpose theo cấu hình riêng. Báo cáo đã loại lần chạy sai preprocessing. Sau khi đánh giá lại đúng cách, full5 trên Clean tái lập các chỉ số cũ (FNR 5%, FPR 30%), còn kết quả suy giảm v2 được báo riêng dưới đây. Bộ ảnh sạch không bị ghi đè.

| Mô hình | Tập test | FNR (95% CI) | FPR | Recall bent / loose / missing / pierced | Review | Chi phí / 1.000 hộp |
|---|---|---:|---:|---|---:|---:|
| Cascade 2 tầng | Clean | 0,00% [0,00–1,58] | 0,00% | 0 / 31,67 / 38,33 / 8,33% | 83% | 9.893,75 |
| Cascade 2 tầng | Degraded v2 | 0,00% [0,00–1,58] | 0,00% | 0 / 5 / 8,33 / 0% | 97% | 9.981,25 |
| Full5 | Clean | 5,00% [2,88–8,53] | 30,00% | 68,33 / 95 / 100 / 65% | 0% | 535,00 |
| Full5 | Degraded v2 | 4,58% [2,58–8,02] | 48,33% | 63,33 / 50 / 100 / 83,33% | 0% | 688,33 |

Cascade đạt FNR 0% trên mẫu test nhưng đánh đổi bằng review 83–97%, khiến chi phí mô phỏng gần 10.000 đơn vị/1.000 hộp theo ma trận đã khai báo; không xem đây là cấu hình triển khai đạt yêu cầu. Chi phí là ước lượng theo prior lỗi 5% và ma trận 100:10:1, không phải chi phí tiền tệ thực tế.

## 5. Kết quả hiện có

### 5.1 Kết quả test khóa của B0–B6

| Phiên bản | Macro-F1 test, mean ± SD (%) | Accuracy test, mean ± SD (%) | FNR, mean ± SD (%) | FPR, mean ± SD (%) |
|---|---:|---:|---:|---:|
| B0 | 97,64 ± 0,67 | 97,66 ± 0,67 | 1,80 ± 0,24 | 1,15 ± 1,00 |
| B1 | **99,34 ± 0,00** | **99,33 ± 0,00** | **0,41 ± 0,00** | **0,00 ± 0,00** |
| B2 | 98,43 ± 0,86 | 98,44 ± 0,84 | 0,55 ± 0,24 | 0,57 ± 1,00 |
| B3 | 97,31 ± 0,00 | 97,32 ± 0,00 | 1,66 ± 0,41 | 0,00 ± 0,00 |
| B4 | 97,55 ± 1,07 | 97,55 ± 1,08 | 1,94 ± 1,57 | 1,15 ± 1,00 |
| B5 | 97,98 ± 1,21 | 97,99 ± 1,21 | 1,11 ± 1,20 | 0,00 ± 0,00 |
| B6 | 97,99 ± 0,89 | 97,99 ± 0,88 | 1,24 ± 0,72 | 1,72 ± 0,00 |

![Hình 5.1. So sánh macro-F1 trên tập test khóa của bảy cấu hình B0–B6](docs/assets/straw_overview/confirmatory_macro_f1.png)

Mỗi giá trị là trung bình và độ lệch chuẩn giữa ba seed trên cùng tập test khóa gồm 299 ảnh (241 lỗi, 58 tốt). Vì test có 299 ảnh, các lần chạy trên cùng ảnh không phải các mẫu độc lập. B1 cho kết quả trung bình cao nhất và ổn định nhất trong bảng; B2 đứng kế tiếp về macro-F1. So sánh chính B6−B1 là -1,35 điểm phần trăm macro-F1, CI 95% theo paired group bootstrap [-2,63; -0,32] (1.000 lần lấy mẫu). Kết quả này là bằng chứng B6 thấp hơn B1 trong phạm vi dataset/protocol này; không chứng minh B1 sẽ tốt hơn trên nhà máy hay camera khác.

| Phiên bản | Latency end-to-end p50 (ms) | p95 (ms) | Tham số |
|---|---:|---:|---:|
| B0 | 167,64 | 181,27 | 11.179.077 |
| B1 | 175,95 | 185,78 | 11.179.077 |
| B2 | 248,30 | 259,66 | 11.179.077 |
| B3 | 247,19 | 261,22 | 11.181.637 |
| B4 | 247,36 | 255,74 | 11.247.429 |
| B5 | 246,24 | 255,65 | 11.312.965 |
| B6 | 244,74 | 257,26 | 11.313.477 |

Latency gồm decode ảnh, tạo view, chuẩn hóa, transfer và forward; batch 1, warmup 5 lượt, 30 lượt đo, GPU Tesla T4. Đây là benchmark của pipeline nghiên cứu PyTorch, không phải latency TensorRT trên demo băng chuyền. B0 có p50 thấp nhất; các cấu hình nhiều view có latency end-to-end cao hơn đáng kể.

Artifact gốc: `straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1/report/test_summary.csv`, `test_runs.csv`, `benchmarks.csv`; từng run có `test.json` và prediction đã được freeze.

### 5.2 Pilot B0, B1 và B5

| Mô hình | Mô tả | Seed | Epoch chạy | Macro-F1 Val | Accuracy Val | FN lỗi→good | FP good→lỗi | Nhầm loại lỗi |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | Global 224 | 42 | 3 | 0,96717 | 0,96667 | 5 | 0 | 5 |
| B1 | Global 512 | 42 | 3 | 0,98076 | 0,98000 | 3 | 0 | 3 |
| B5 | Global + local attention có ngữ cảnh | 42 | 3 | 0,95713 | 0,95667 | 4 | 2 | 7 |

Đây là pilot ba epoch, seed 42, chỉ dùng Train/Validation; không thay thế kết quả xác nhận và test ở Mục 5.1. Trong confirmatory run, B1 đã được chọn làm baseline theo validation trước khi mở test.

### 5.3 Độ sẵn sàng

Artifact readiness ghi `READY_FOR_INTERNAL_THESIS_ANALYSIS`, đủ 21 run, test output, freeze và benchmark. Tuy nhiên, trong `freeze.json`, trường reviewer vẫn là văn bản mẫu “Tên người thực sự đã kiểm tra” và notes vẫn là hướng dẫn mẫu. Vì vậy, dù số liệu test B0–B6 đầy đủ, không thể xem artifact hiện tại là bằng chứng đã hoàn tất rà soát thủ công nhãn/nhóm/quyền dữ liệu/related work cho bản nộp cho đến khi thay bằng thông tin và biên bản thật. Readiness kỹ thuật không phải phê duyệt của trường hoặc bằng chứng novelty.

### 5.4 Bằng chứng triển khai

| Thành phần | Bằng chứng | Diễn giải hợp lệ |
|---|---|---|
| TensorRT/DeepStream replay | 300 frame, macro-F1 99,016%, accuracy 99,000% | Validation replay của checkpoint nhánh cũ |
| RTSP replay | 252 frame, 31,03 FPS, first prediction 1,31 s | Pipeline replay hoạt động ở cấu hình ghi nhận |
| TensorRT resident worker | Engine/context/buffer nạp một lần, warmup ba lượt | Model ở trong bộ nhớ và xử lý theo request |
| Pixel gate canvas | Ba sự kiện/ba suy luận; unit test đạt | Luồng sự kiện hoạt động ở mô phỏng nhỏ |
| 30 ảnh Bent suy giảm trên băng chuyền canvas | 18/30 đúng lớp; ảnh input bị xoay EXIF 90°; demo mặc định là B1 384 | Không dùng làm độ chính xác B1 512; cần tạo lại đúng chiều và chạy đúng artifact |
| Test Factory Degraded v2, full5 | FNR 4,58%; FPR 48,33%; recall Bent 63,33%; review 0% | 300 ảnh đã sửa tên; checkpoint triển khai cũ; cần lưu ý hiệu năng giảm và false positive cao |
| Test sạch B0–B6 | B1 cao nhất: macro-F1 99,34%; B6: 97,99% | Test nội bộ 299 ảnh × 3 seed; khác checkpoint và phân phối với demo degraded |
| Test cascade v2 | FNR 0%; review 97% degraded và 83% clean | Bỏ sót thấp đi kèm tỷ lệ review/chi phí không phù hợp |
| Bảng Factory Degraded cũ | Không dùng để kết luận | Bộ sinh ảnh ghi đè giữa các thương hiệu; tiền xử lý cũ cần khớp EXIF |
| Gate trên camera | Chưa có video có nhãn | Chưa thể kết luận recall, false trigger hay bỏ sót |

## 6. Đánh giá, giới hạn và rủi ro nghiên cứu

### 6.1 Điểm mạnh

- Bài toán và đầu ra được giới hạn rõ; thương hiệu không làm nhãn hoặc điều kiện dự đoán.
- B0–B6 tạo đối chứng theo độ phân giải, global/local, pooling, attention, ngữ cảnh và tọa độ.
- Split có group, SHA và fingerprint; test được bảo vệ bởi runner; resume hỗ trợ tái lập.
- Chỉ số phân biệt bỏ sót sản phẩm lỗi với nhầm loại lỗi, đồng thời yêu cầu báo chi phí.
- Nguyên mẫu TensorRT và gate pixel minh họa một đường triển khai vượt khỏi notebook huấn luyện.

### 6.2 Việc còn thiếu

1. Xác minh định nghĩa năm nhãn gốc và rà mẫu khó, đặc biệt `loose`/`pierced`.
2. Rà nhóm ảnh gần nhau; lưu người rà, quyết định và căn cứ. Hash không thay thế metadata phiên chụp.
3. Ghi nguồn tải, quyền sử dụng và điều kiện chia sẻ dataset.
4. Hoàn tất rà soát công trình liên quan và bảng khác biệt; không suy ra tính mới từ tên global-local.
5. Thay placeholder người rà soát/ghi chú trong freeze ledger bằng thông tin, phạm vi và biên bản rà soát thực tế trước khi dùng làm bằng chứng nộp.
6. Bổ sung bootstrap theo group cho các so sánh còn lại, phân tích lỗi ảnh và benchmark đầy đủ trên cùng môi trường.
7. Tạo lại tập suy giảm cho B1 sau EXIF transpose; chạy một lượt với engine có SHA xác nhận và cùng độ phân giải 512; không dùng lô cũ xoay ngang làm robustness score.
8. Thu video camera có nhãn sự kiện để đánh giá gate; đo nhiều tốc độ, ánh sáng, bóng, che khuất và khoảng cách sản phẩm.

### 6.3 Giới hạn diễn giải

Ba seed không phải ba test set độc lập. Tập test confirmatory có 299 ảnh, nên độ chính xác ước lượng còn hạn chế, nhất là tỷ lệ lỗi thấp. Dataset đã được dùng trong phát triển nên kết quả không tương đương holdout bên ngoài mới. Freeze artifact còn placeholder cho người rà soát; cần hoàn tất hồ sơ trước khi viện dẫn readiness như xác nhận kiểm tra thủ công. Không có nhãn tọa độ nên attention không chứng minh định vị. RTSP và ROI mô phỏng không kiểm chứng camera thật, PLC, throughput sản xuất hay độ bền dài hạn. Lô 30 ảnh từng đạt 60% bị xoay 90° bởi preprocessing EXIF và dùng artifact khác với bảng test B1 512, nên chưa đo được độ bền B1 trên ảnh suy giảm đúng chiều.

### 6.4 Không trộn kết quả B0–B6 khác định nghĩa

Trong workspace có kết quả của nhánh phân loại trước. Mã B0–B6 ở báo cáo này có định nghĩa tại Mục 2.4 và dùng protocol/fingerprint hiện tại. Chỉ nhập bảng cũ nếu xác minh được cấu hình, checkpoint, split, class order, preprocessing và sample ID tương ứng. Nếu chưa khớp, trình bày riêng như kết quả lịch sử; không gộp trung bình hoặc dùng làm test của thí nghiệm mới.

## 7. Kế hoạch hoàn thiện và tiêu chí kết luận

### 7.1 Thứ tự triển khai

**Giai đoạn 1 — khóa dữ liệu và nền tảng:** chuẩn hóa định nghĩa lớp; rà group; ghi provenance/quyền; hoàn thiện related work; cố định protocol và cấu hình.

**Giai đoạn 2 — pilot phát triển:** hoàn tất pilot đủ ngân sách trên Train/Validation, bao gồm B0/B1 và global-local đại diện. Không dùng Test để quyết định thay đổi mô hình.

**Giai đoạn 3 — xác nhận (đã hoàn tất):** B0–B6 đã chạy đủ ba seed; checkpoint, log và fingerprint có trong artifact confirmatory.

**Giai đoạn 4 — freeze và đánh giá (đã hoàn tất cho B0–B6):** selection được khóa theo Validation trước test; 21 test output được xác minh. Bảng mean ± SD và so sánh paired chính đã có. Cần thay thông tin review placeholder trong ledger và bổ sung bootstrap/ma trận lỗi cho các so sánh phụ trước khi nộp.

**Giai đoạn 5 — triển khai:** chọn checkpoint đã freeze; kiểm tra parity preprocessing; benchmark batch 1 p50/p95; so sánh PyTorch/TensorRT cùng đầu vào; báo startup riêng với steady-state.

**Giai đoạn 6 — kiểm chứng gate:** tạo video ngắn theo sản phẩm, gán ID/class/thời điểm vào ROI; đo event recall, false trigger/phút, số lần gọi AI/sản phẩm, accuracy/FNR sau gate và latency từ ROI đến kết quả. Chạy shadow mode trước khi bỏ frame khỏi nhánh inference.

### 7.2 Quy tắc kết luận

- Nếu global-local cải thiện ổn định so với baseline mạnh nhất theo validation và test khóa, CI phù hợp, và chi phí chấp nhận được, báo bằng chứng hỗ trợ cải thiện trong phạm vi dataset.
- Trong kết quả hiện có, B1 đạt macro-F1 cao hơn B6; chênh lệch paired B6−B1 âm với CI 95% không chứa 0. Chưa có bằng chứng rằng cấu hình đề xuất B6 cải thiện baseline B1 trên protocol này.
- Nếu CI chứa 0 hoặc hiệu ứng thay đổi mạnh theo seed, kết luận chưa có bằng chứng rõ; không viết “tương đương” nếu chưa có kiểm định tương đương.
- Nếu baseline toàn ảnh thắng, kết luận baseline phù hợp hơn với dữ liệu này và phân tích lý do.
- Nếu B6 không hơn B5 thì chưa có bằng chứng vị trí giúp ích; nếu B5 không hơn B4 thì chưa có bằng chứng ngữ cảnh global có đóng góp.
- Nếu TensorRT đổi kết quả so với nguồn chuẩn, nêu sai khác preprocessing/backend và không fine-tune trên Test.

### 7.3 Bằng chứng hoàn thành cần có

| Nhóm | Điều kiện tối thiểu |
|---|---|
| Dữ liệu | Quy tắc nhãn, rà group, nguồn/quyền, manifest và hash được lưu |
| Nghiên cứu | Related work và claim novelty được giới hạn theo đối chứng thực tế |
| Mô hình | Baseline 224/512 và ablation đăng ký, đủ seed hoặc thu hẹp công khai |
| Test | Freeze trước test, predictions khớp sample ID, confusion/per-class và bất định |
| Chi phí | Decode/preprocess/forward/API tách riêng, p50/p95, VRAM/parameters/GPU |
| Phân tích | Lỗi đại diện, trả lời từng RQ, giới hạn và kết quả âm |
| Demo | Preprocessing khớp model; video gate; không nhận diện đạt khi chưa inference |

## 8. Kết luận

Dự án đã hoàn tất 21 run xác nhận B0–B6 trên ba seed và có test khóa 299 ảnh. B1 512 đạt macro-F1 99,34% ± 0,00; B6 đạt 97,99% ± 0,89. So sánh paired B6−B1 là -1,35 điểm phần trăm (CI 95% [-2,63; -0,32]), nên các kết quả hiện tại không ủng hộ giả thuyết B6 cải thiện baseline B1 trong phạm vi protocol này. Benchmark cho thấy B0 nhanh nhất p50; các cấu hình nhiều view mất thêm thời gian.

Kết quả sạch không dự báo được ảnh suy giảm: full5 trên degraded v2 có FPR 48,33%, còn cascade đạt FNR 0% nhưng review 83% clean và 97% degraded. Lô 30 ảnh Bent từng báo đúng 60% bị xoay 90° do EXIF bị bỏ và app mặc định dùng artifact B1 384 thay cho B1 512; không diễn giải đó là kết quả robustness hợp lệ. Trước khi dùng làm bằng chứng nộp, cần chạy lại degraded B1 sau EXIF transpose, xác nhận artifact 512 bằng SHA, thay placeholder người rà soát trong freeze ledger bằng hồ sơ thật và kiểm chứng gate trên video camera. Kết luận nghiên cứu hiện có là B1 thắng B6 trên test nội bộ đã chạy; khả năng vận hành băng chuyền chưa được xác nhận.

## Tài liệu tham khảo

1. He, K., Zhang, X., Ren, S., Sun, J. “Deep Residual Learning for Image Recognition.” Proceedings of CVPR, 2016. DOI: 10.1109/CVPR.2016.90.
2. Ilse, M., Tomczak, J., Welling, M. “Attention-based Deep Multiple Instance Learning.” Proceedings of ICML, 2018, PMLR 80, pp. 2127–2136.
3. Hu, T. et al. “See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification.” ICCV, 2019.
4. PyTorch, TorchVision, NVIDIA TensorRT và NVIDIA DeepStream documentation; phiên bản cần ghi theo môi trường benchmark cuối.
5. Tài liệu nội bộ: `docs/STRAW_RESEARCH_PLAN.md`, `docs/STRAW_GLOBAL_LOCAL_RUNBOOK.md`, `docs/STRAW_GLOBAL_LOCAL_VERIFICATION.md`, `docs/STRAW_COMPLETION_REQUIREMENTS.md`, `docs/STRAW_RTSP_DEEPSTREAM.md`, `docs/STRAW_INTELLIGENT_INFERENCE_GATE.md`.

## Phụ lục A. Luồng xử lý nguyên mẫu băng chuyền

```text
Chọn nhiều ảnh
      ↓
Tạo job và đưa từng ảnh vào hàng đợi mô phỏng
      ↓
Frame canvas → luma → bù sáng → trừ nền → mask tiền cảnh
      ↓
Closing / thành phần liên thông / ghép track theo tâm
      ↓
≥3 quan sát + đi phải sang trái + cắt vạch + ≥90% trong ROI
      ↓
Một sự kiện mỗi track → TensorRT worker thường trú
      ↓
Ảnh sản phẩm gốc → resize theo engine → xác suất năm lớp → gắn về track
      ↓
Trở lại dò pixel để chờ sản phẩm tiếp theo
```

Đây là bản canvas mô phỏng. Camera RTSP thật, đồng bộ PLC, tách sản phẩm sát nhau và chọn frame theo độ nét chưa được xác nhận.

## Phụ lục B. Artifact tái lập

- Protocol/manifest: `artifacts/straw_global_local/data_v1/`.
- Config 21 run: `configs/straw_global_local.json`.
- Config pilot: `configs/straw_global_local_pilot.json`, `configs/straw_global_local_pilot_highres.json`.
- Mô hình/runner: `src/straw_global_local/model.py`, `src/straw_global_local/runner.py`.
- Pilot: `artifacts/verification/global_local_pilot/validation_pilot_summary.csv`.
- Bằng chứng gate: `artifacts/b1_stream/roi_browser_smoke.json`, `artifacts/b1_stream/roi_api_smoke.json`.
- Report DeepStream/RTSP: `artifacts/b1_stream/REPORT.md`.
- Gate/worker: `apps/stream_demo/conveyor_gate.js`, `apps/stream_demo/resident_worker.py`, `apps/stream_demo/resident_inference.py`.
- Kiểm thử gate: `node tests/test_conveyor_gate.cjs`.

## Phụ lục C. Nội dung cần bổ sung trước khi nộp

- Tên trường, khoa, ngành, giảng viên hướng dẫn, sinh viên, mã số, địa điểm và năm.
- Nguồn dataset, giấy phép/điều khoản, quy trình chụp và tiêu chuẩn gán nhãn.
- Biên bản rà nhãn và nhóm ảnh; chỉ điền người rà sau khi việc rà thực sự diễn ra.
- Related work hoàn chỉnh và xác nhận phạm vi novelty với giảng viên.
- Bảng kết quả xác nhận sau freeze, test, bootstrap và benchmark.
- Biểu đồ học, confusion matrix, trade-off hiệu năng–latency và ảnh lỗi đại diện.
- Kết quả video gate nếu giữ phần này như một tuyên bố trong luận văn.
