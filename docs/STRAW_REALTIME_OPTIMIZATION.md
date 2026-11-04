# Nghiên cứu tối ưu B1 cho ứng dụng web kiểm tra lỗi ống hút

> **Cập nhật 21/10/2026:** Web mặc định vẫn dùng cấu hình 384. B1 512 và ngưỡng Good/lỗi là nhánh thử nghiệm; ngưỡng chỉ đổi bỏ sót thành sai loại, chưa giảm tổng lỗi validation. [Báo cáo so sánh và tiêu chí trước triển khai](../straw_resume/error_analysis/B1_REALTIME_UPGRADE.md).

## Mục tiêu và phạm vi kết luận

Giữ B0–B6 làm nghiên cứu so sánh biểu diễn. Phần mở rộng nghiên cứu hệ thống B1: giảm chi phí truyền ảnh, tiền xử lý và suy luận, sau đó phục hồi chất lượng khi giảm độ phân giải hoặc lượng tử hóa gây suy giảm. Cần báo cáo cả các phương án thất bại; nén không mặc nhiên làm mô hình nhanh hơn.

Đây là nghiên cứu ứng dụng có kiểm chứng trên một bộ dữ liệu và laptop, chưa phải phương pháp nén mới hoặc hệ thống đã xác nhận cho dây chuyền sản xuất. Tập test cũ đã được xem trong quá trình phát triển: các kết quả mới là mở rộng sau phân tích, không trình bày như một kiểm định độc lập được đăng ký trước. Không thay split để tìm điểm số cao hơn.

## Bài toán tối ưu có ràng buộc

Gọi c là cấu hình gồm kích thước ảnh, JPEG, runtime, precision và chính sách xử lý yêu cầu. Mục tiêu chọn c trên validation và phép đo hệ thống:

\[
\min_c L_{95}(c), \quad
F_1(c) \ge F_1(c_0)-\delta,\quad
FNR(c)\le FNR(c_0)+\epsilon_{FN},\quad
FPR(c)\le FPR(c_0)+\epsilon_{FP},\quad
M(c)\le M_{budget}.
\]

c0 là B1 FP32 512 với tiền xử lý gốc. Tiêu chí triển khai thận trọng ban đầu: giảm Macro-F1 không quá 0,5 điểm phần trăm, không tăng số FN/FP trên validation. Đây là tiêu chí kỹ thuật đề xuất, không phải bảo đảm thống kê hoặc yêu cầu từ nhà máy. Với 240 ảnh lỗi và 60 ảnh tốt ở validation, thêm một FN thay đổi FNR khoảng 0,417 điểm phần trăm; thêm một FP thay đổi FPR khoảng 1,667 điểm phần trăm. Không báo nhiều chữ số như thể đã có độ chính xác thống kê tương ứng.

Giữ các cấu hình trên biên Pareto: không có cấu hình khác đồng thời chính xác hơn, nhanh hơn và ít bộ nhớ hơn. Không gộp mọi chỉ số vào một điểm trọng số tùy ý để tuyên bố một cấu hình tốt nhất cho mọi hoàn cảnh.

## Tách ngân sách độ trễ

\[
L_{browser}=L_{capture}+L_{resize/encode}+L_{upload}+L_{request/server}+L_{download}+L_{render}.
\]

\[
L_{server}=L_{body/parse}+L_{queue}+L_{decode/normalize}+L_{transfer/infer}+L_{postprocess}+L_{serialize}.
\]

Client dùng performance.now từ trước drawImage/toBlob đến khi nhận JSON và ghi thêm thời điểm DOM được cập nhật. Server dùng đồng hồ riêng để phân rã thời gian; không trừ hai timestamp tuyệt đối trên hai máy. server_request hiện đo từ middleware đến trước trả kết quả; chưa bao gồm toàn bộ serialize và truyền response. Số đo HTTP phía client mới bao phủ vòng request/response. GPU phải đồng bộ đúng khi đo; trả logits về CPU trong runtime đã đồng bộ đầu ra.

Mục tiêu thử nghiệm: p95 capture→hiển thị dưới 200 ms với một client, tối thiểu 5 kết quả/giây, 30 phút không tăng hàng đợi hoặc rò bộ nhớ. Đây là mục tiêu, chỉ đánh dấu đạt sau đo. Báo thêm p99, max, deadline-miss rate, request bị từ chối và số frame bỏ; FPS video preview khác FPS phân tích.

## 1. Tiền xử lý và truyền ảnh

- Khởi đầu giữ nguyên RGB, EXIF, resize vuông PIL bilinear và chuẩn hóa ImageNet của B1. Viết kiểm tra tương đương tensor trước tối ưu. Không dùng nhầm BGR hoặc normalize hai lần.
- Đưa resize về trình duyệt trước upload cho chế độ camera. Giữ preview riêng; gửi ảnh JPEG 512 px quality 0,9 bằng Blob/multipart. Không gửi base64 và ảnh 3000×3000 cho mỗi lần dự đoán.
- So sánh nguyên bản với JPEG 512 và JPEG 384 cùng quality 90. Chỉ thử thêm quality 80 khi 90 còn tốn băng thông đáng kể. PIL JPEG trong script là mô phỏng offline; browser canvas có nội suy/encoder khác, nên phải kiểm tra lại ảnh do trình duyệt thực sự tạo.
- Giữ so sánh chất lượng theo cùng ảnh. Kích thước/crop phải độc lập nhãn. Không crop chỉ vùng ống hút bằng tên lớp hoặc thương hiệu. Nếu camera yêu cầu ROI cố định, khảo sát riêng vì có thể làm mất bằng chứng của lỗi Missing/Loose.
- Tự động tăng tương phản, sharpen, CLAHE, tách nền hoặc mask giả chưa có bằng chứng giúp dữ liệu này. Chỉ thêm như ablation có kiểm chứng nếu có lỗi ánh sáng cụ thể.
- Đo client resize/encode và upload riêng: chuyển công việc từ server sang browser không đồng nghĩa loại bỏ công việc.

## 2. Thực thi và nén mô hình

Các phép thử đầu tiên: B1 PyTorch GPU FP32; FP16; FP16 + channels-last; ONNX FP32 CPU; ONNX INT8 PTQ CPU. Bật autotune cuDNN với shape cố định trong triển khai, lưu cấu hình để tái lập. Đánh giá lại chất lượng sau mọi thay đổi số học. FP16 có thể chậm hơn FP32 trên workload nhỏ hoặc do kernel/layout; không chọn dựa trên tên precision.

Export ONNX phải dùng checkpoint đã huấn luyện, batch=1, shape cố định. Kiểm tra ONNX và sai khác logits trước khi đánh giá toàn validation. Chuyển file sang ONNX không phải bằng chứng tăng tốc. Đồ thị cần được thực thi bằng provider thật sự có trên máy.

INT8 PTQ: dùng static QDQ, trọng số per-channel và 200 ảnh train cân bằng 40 ảnh/lớp để hiệu chuẩn. Ghi sample IDs. Không dùng test để hiệu chuẩn. Vì đây là CNN, static là lựa chọn ban đầu phù hợp hơn dynamic. Benchmark INT8 trên CPU, không coi CUDA provider thông thường tự động tăng tốc mọi toán tử INT8. Máy hiện có ONNX Runtime CPU; TensorRT chưa cài và chưa được đo.

Lượng tử hóa tuyến tính:

\[
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),\quad
\hat{x}=s(q-z).
\]

Phải cân bằng sai số làm tròn với clipping. Nếu INT8 suy giảm: kiểm tra preprocess, calibration distribution, per-channel, lớp nhạy và sai khác activation trước; sau đó mới QAT. ZIP nhỏ hơn chỉ giúp lưu trữ/truyền file, không giảm FLOPs. Pruning đặt trọng số bằng 0 nhưng giữ nguyên tensor cũng không bảo đảm giảm latency. Structured pruning chỉ bổ sung nếu thực sự cần thêm một nhánh nghiên cứu và đã đo được lợi ích của backbone nhỏ trên runtime đích.

## 3. Phục hồi bằng huấn luyện đúng nguyên nhân

Nếu JPEG/384 làm mất chất lượng: fine-tune từ đúng checkpoint seed tương ứng với cùng đường tiền xử lý triển khai. Dùng train gốc, cross-entropy, lr 1e-5, batch=2, tích lũy tối đa 4 batch, 5 epoch làm pilot; giữ BN frozen như mô hình gốc, brightness/flip giống giả định dữ liệu. Chọn checkpoint bằng validation, sau đó export lại. Không bảo đảm phục hồi được chi tiết vốn đã bị mất ở 384 px.

Nếu INT8 PTQ không đạt tiêu chí: QAT từ B1, dùng fake quantization ở train và đánh giá mô hình INT8 đã convert thật trên CPU sau mỗi epoch. Không chọn theo accuracy mô hình float/fake-quant rồi giả định INT8 giống vậy. QAT tối ưu:

\[
\min_\theta \frac1N\sum_i CE(f_{Q(\theta)}(T_{deploy}(x_i)),y_i).
\]

Gradient xấp xỉ STE cho bước làm tròn; observer thống kê miền giá trị, sau đó đóng băng observer và BN trong giai đoạn cuối. QAT không đồng nghĩa huấn luyện lại từ đầu hoặc chưng cất. Không cần teacher trong phạm vi này.

`recover_b1_deployment.py` hỗ trợ finetune/QAT, lưu checkpoint mỗi epoch và RNG/optimizer để tiếp tục. QAT xuất `best_int8.ts` cho CPU PyTorch; đây không phải ONNX/TensorRT INT8. Cần đánh giá runtime phù hợp với định dạng xuất. Smoke test QAT chỉ kiểm tra chu trình kỹ thuật, không tính thành kết quả chất lượng.

Pilot chỉ dùng seed 42. Chốt một nhánh có triển vọng rồi kiểm chứng bằng seed 43/44; mỗi seed khởi tạo từ checkpoint tương ứng. Không chạy ma trận hàng chục cấu hình và ba seed cho tất cả trước khi biết nút thắt.

## 4. Hậu xử lý

Đầu tiên giữ argmax 5 lớp để so sánh công bằng với baseline. Softmax thực hiện bằng số thực ổn định; không cần NMS/morphology vì đầu ra là nhãn ảnh, không phải bounding box/mask.

Nếu cần điểm xác suất có ý nghĩa, nghiên cứu temperature scaling: T*=argmin NLL(z/T,y). Phải tách nhóm ảnh dev thành phần fit nhiệt độ/ngưỡng và phần đánh giá chính sách hoặc dùng group cross-fitting. Không fit rồi báo hiệu quả trên chính các ảnh đó như đánh giá độc lập. Đo NLL, Brier, reliability diagram; ECE chỉ là mô tả trên bộ nhỏ. Temperature dương không thay đổi argmax nên không tự sửa nhãn sai.

Ngưỡng nhị phân dùng s_defect=1-p_good. Chọn tau bằng dev dưới ràng buộc FN, báo risk-coverage nếu có trạng thái kiểm tra lại. Không tự động gọi mẫu bị từ chối là dự đoán đúng. Không tăng ngưỡng chỉ để giảm FP nếu làm bỏ sót lỗi tăng. Baseline hiện không có FP trên validation/test cũ, nên không thể hứa cải thiện FP ở đó.

Làm mượt theo thời gian chỉ nghiên cứu khi biết nhiều frame thuộc cùng sản phẩm và có ranh giới sản phẩm. Khi đó có thể khảo sát EMA p_t=alpha*p_t+(1-alpha)*p_(t-1), reset theo product_id, báo thêm độ trễ phát hiện và recall sự kiện. Dataset ảnh tĩnh hiện tại không chứng minh được hiệu quả temporal smoothing; không tạo video lặp ảnh rồi coi đó là bằng chứng triển khai thực tế.

## 5. Kiến trúc web và điều tiết tải

Một tiến trình/worker GPU, một model nạp lúc khởi động, warm-up trước sẵn sàng. Chế độ camera hiện dùng request nối tiếp: chụp frame mới sau khi request trước hoàn thành; không tạo backlog ở client. Server cho phép một yêu cầu chờ tối đa 50 ms, trả 429 khi đầy/hết ngân sách để client gửi frame mới sau. Trả frame_id để loại phản hồi cũ khi dừng/khởi động camera; hiển thị riêng snapshot đã phân tích.

Chỉ trả JSON nhãn/điểm/thời gian. Tính giải thích, lưu ảnh, truy xuất ví dụ và tạo báo cáo ở luồng riêng theo yêu cầu. Không chạy hai model 224 của demo cũ rồi gán điểm số B1 512 cho chúng. Không khởi tạo random weights nếu checkpoint thiếu.

WebSocket là một biến thể giao vận có thể thử sau khi xác định HTTP overhead đáng kể; đổi giao thức không sửa được model chậm hoặc queue vô hạn. Với một client localhost, HTTP multipart nối tiếp đủ làm baseline có thể đo và kiểm chứng.

Chế độ bỏ frame phù hợp live preview. Dây chuyền bắt buộc kiểm tra từng sản phẩm cần trigger/tracking, hàng đợi có hạn và báo quá tải; không được lặng lẽ bỏ sản phẩm.

## 6. Ma trận thực nghiệm và báo cáo

| Nhóm | Cấu hình chính | Câu hỏi |
|---|---|---|
| P | Gốc512; JPEG512q90; JPEG384q90 | Giảm ảnh truyền có làm mất chất lượng? |
| R | GPU FP32; FP16; FP16 channels-last | Precision/layout nào nhanh nhất thực tế? |
| C | ONNX FP32 CPU; ONNX INT8 CPU | Chuyển đổi/nén cải thiện tài nguyên đến đâu? |
| F | 384 trước/sau fine-tune nếu P không đạt | Phục hồi chất lượng được bao nhiêu? |
| Q | PTQ; QAT INT8 nếu PTQ không đạt | QAT có giúp vượt giới hạn PTQ? |
| W | Một client, hai client; camera; ảnh upload | Tuổi kết quả, hàng đợi và tải liên tục? |

Mỗi cấu hình cần: Macro-F1, per-class recall, FN/FP/sai loại, NLL khi phù hợp; kích thước artifact thực tế; p50/p95/p99 wall-clock; RAM/VRAM của tiến trình triển khai độc lập; số frame/request, 429, drop, lỗi và deadline-miss. Không dùng RSS của tiến trình đang giữ nhiều mô hình làm bộ nhớ một mô hình.

Benchmark khởi động và steady-state tách riêng. Dùng 20 ảnh validation cân bằng làm workload độ trễ, 20 warmup và ít nhất 100–1000 lượt cho xác nhận cuối. Pilot script dùng 60 lượt để sàng lọc, không đủ thay bài stress 30 phút. Dùng cùng đầu vào/thời gian nghỉ/nguồn điện cho so sánh, không chạy training cùng lúc đo. GPU clock/điện/nhiệt và máy dùng tác vụ khác ảnh hưởng đáng kể.

Khóa lựa chọn bằng validation; chạy test sau khi chốt, báo đây là test nội bộ đã từng được xem. Không tái fine-tune hoặc chọn nhánh tiếp dựa trên test rồi vẫn gọi nó là đánh giá độc lập. Kết luận giới hạn trong bộ dữ liệu ảnh này; chưa có unseen-brand/domain hoặc camera thật để khẳng định tổng quát.

## Tập tin và lệnh chạy

- `scripts/research/optimize_b1_deployment.py --phase all`: nghiên cứu validation, ONNX/PTQ và profiling. Chạy bằng `.venv/Scripts/python.exe` từ gốc repo.
- `scripts/research/recover_b1_deployment.py --method finetune --size 384 --epochs 5`: nhánh phục hồi chất lượng ảnh nhỏ; tự tiếp tục khi có last.pt.
- `scripts/research/recover_b1_deployment.py --method qat --size 512 --epochs 5`: nhánh QAT CPU, chỉ chạy khi PTQ cần phục hồi. Thêm `--smoke` cho kiểm tra kỹ thuật.
- `run_b1_web.ps1`: mở server B1 tại `http://127.0.0.1:8032`. Bấm Bật camera trên web để cấp quyền; phần camera thực phải được người dùng kiểm tra.
- `scripts/research/benchmark_b1_web.py --seconds 30 --clients 1 --rate 10`: đo HTTP có nhịp gửi; đổi `--clients 2` để kiểm tra quá tải. Không chạy cùng training. Đổi `--seconds 1800` cho bài stress 30 phút, nhưng vẫn cần kiểm tra camera riêng.
- `scripts/research/select_b1_deployment.py`: chọn cấu hình trong các ứng viên đã đo theo validation và p95, lưu selection.json; phải restart server sau khi đổi config.
- `scripts/research/summarize_b1_web.py`: sau các bài đo HTTP và phát lại trình duyệt, bổ sung bằng chứng vào REPORT.md mà không chạy lại training/profiling. Từ chối ghép bằng chứng khác SHA checkpoint.
- `run_b1_research.ps1 -Recovery`: thực hiện nghiên cứu cơ sở và hai nhánh phục hồi, lưu kết quả; các lượt kiểm chứng `--seed 43/44 --tag safety` được báo riêng.
- `configs/straw_realtime.json`: cấu hình triển khai B1 riêng; không thay demo phân cấp cũ.
- `artifacts/b1_deployment/`: kết quả mới, mô hình chuyển đổi, calibration IDs và checkpoint phục hồi.

## Tài liệu kỹ thuật gốc

- PyTorch Performance Tuning: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime I/O Binding: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
- TensorRT Accuracy Considerations: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html

Các tài liệu này mô tả cơ chế và lựa chọn triển khai; chúng không chứng minh tốc độ/chất lượng cho bộ dữ liệu này. Phần đó phải lấy từ thí nghiệm cục bộ.

## Cấu hình được chọn trong lượt triển khai này

B1 fine-tune 384 px, FP16, channels-last, JPEG q90, seed phục vụ cố định 42. Lưu trọng số FP16 riêng ở `artifacts/b1_deployment/models/b1_web384_fp16.pt`, kiểm tra round-trip các tensor và SHA-256 khi web khởi động. Cấu hình đã chọn và điều kiện chất lượng nằm trong `selection.json`; không chọn seed tốt nhất.

Nhánh 384 sau phục hồi được kiểm tra thêm seed 43/44. Độ trễ các runtime được đo lại khi không còn training. QAT/ONNX INT8 không được chọn nếu chất lượng không đạt tiêu chí. Xem số đo và giới hạn thực tế trong `artifacts/b1_deployment/REPORT.md`.

Web warm-up và xử lý trên cùng một luồng GPU. Có tối đa một yêu cầu đang xử lý và một yêu cầu chờ, ngân sách chờ tổng 50 ms; vượt giới hạn thì trả 429 để client lấy frame mới. Khi client hủy request, future GPU được shield và vẫn giữ trạng thái bận cho đến lúc hoàn thành, tránh tích lũy tác vụ cũ trong executor. Đây là điều tiết tải phục vụ demo, chưa phải bảo đảm hard real-time trên mọi tải/mạng.

## Định hướng triển khai trên phần cứng có NVIDIA GPU

Backend hiện triển khai là PyTorch CUDA FP16, không phải ONNX CPU. Chọn theo số đo trên RTX 3050 Laptop; CPU tiếp tục giải mã ảnh, chuẩn hóa và xử lý JSON. Không ép từng công đoạn nhỏ lên GPU khi chưa chứng minh lợi ích sau chi phí truyền dữ liệu/đồng bộ. GPU có sẵn là một ứng viên cần khai thác, không phải bằng chứng mọi cấu hình GPU đều nhanh hơn mọi cấu hình CPU.

Đối chiếu engine cần PyTorch CUDA FP32/FP16 với TensorRT FP16 trên cùng GPU, cùng checkpoint, đầu vào, batch, nhịp tải và phạm vi đồng hồ. Nhánh RTSP/DeepStream đã chạy TensorRT FP16 trên H.264, có báo cáo riêng tại `artifacts/b1_stream/REPORT.md` và thiết kế tại `docs/STRAW_RTSP_DEEPSTREAM.md`; đây chưa phải so sánh engine trên cùng tensor với ma trận JPEG của web. ONNX Runtime CUDA vẫn chưa đo. Kết quả ONNX INT8 CPU hiện tại không chứng minh INT8/TensorRT GPU chậm; chưa gọi cấu hình nào tối ưu tuyệt đối. Theo [NVIDIA](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html), thiết lập baseline, đo rồi tối ưu và đo lại; chú ý truyền PCIe, nhiệt/clock và bộ nhớ khi so sánh.

Batch=1 được giữ cho đường camera đơn có yêu cầu phản hồi nhanh. Khi nhiều camera, chỉ thử gom batch nếu độ trễ chờ vẫn nằm trong ngân sách; báo throughput cùng p95/p99 và số yêu cầu trễ/từ chối. Không dùng FPS suy luận kernel làm FPS kiểm tra sản phẩm.

Phát lại validation thực trên trình duyệt được chuẩn bị bằng `prepare_browser_replay.py`, sau đó mở `/validation-check` trên server. 300 ảnh nguồn proxy 640 px đi qua canvas/JPEG/API/UI thật; server đối chiếu sample ID và nhãn từ manifest, lưu `browser_replay.json`. Đây là phép kiểm tra tương thích tiền xử lý và độ trễ web trên ảnh tĩnh, không thay kiểm tra cảm biến, chuyển động, nhòe hoặc ánh sáng camera. Báo riêng kết quả chất lượng và latency của phép phát lại; không gán số đo offline PIL cho canvas trình duyệt.

Khi chạy thật, cần xác định tốc độ dây chuyền và thời hạn ra quyết định cho từng sản phẩm. Cơ chế bỏ frame preview không bảo đảm kiểm tra đủ sản phẩm; cần trigger/product_id, xử lý quá tải và theo dõi lỗi rõ ràng. Local demo chưa triển khai cơ cấu loại sản phẩm hoặc kiểm chứng bền bỉ 30 phút. Các route phục vụ ảnh validation chỉ dùng cục bộ để nghiên cứu; không công khai nguyên trạng khi đóng gói dịch vụ bên ngoài.
