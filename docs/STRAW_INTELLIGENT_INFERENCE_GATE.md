# Cổng suy luận truyền thống trước CNN

## Trạng thái triển khai ngày 21/10/2026

Demo chọn ảnh ở cổng 8033 đã triển khai gate thật trên pixel **cảnh mô phỏng**, chưa phải gate trên RTSP camera thật. Chọn ảnh tự thêm vào hàng đợi. `conveyor_gate.js` nhận ảnh nền trống và RGBA của frame, không nhận tọa độ sản phẩm hay nhãn: chuyển luma, bù sáng toàn cảnh, trừ nền, closing 3×3, thành phần liên thông, ghép track theo tâm; xác nhận ≥3 quan sát, chiều đi từ phải sang trái, cắt đường giữa và ≥90% bbox nằm trong ROI mới phát một sự kiện. Băng tải chuyển động và lớp nhãn/ROI không nằm trong ảnh phân tích. Track đã kích hoạt không được gọi lần hai.

`resident_worker.py` nạp engine TensorRT FP16 đã có trên RTX 3050, tạo context, cấp phát GPU/pinned-host buffers một lần và warm-up ba lượt; sau đó chỉ xử lý queue khi có sự kiện ROI. Đây là TensorRT native trong container NVIDIA hiện có, không dựng video/khởi động `nvinfer` lại theo mỗi lượt chọn ảnh. Khi không có sản phẩm, CPU tiếp tục phát hiện còn GPU không có lượt phân loại sản phẩm mới. Đầu vào là ảnh sản phẩm gốc đầy đủ resize 384×384 RGB/CHW float32 0..255; normalization đã nằm trong engine. Chuỗi tiền xử lý trực tiếp này khác đường RTSP/H.264/YUV cũ, vì vậy cần kiểm chứng chất lượng riêng trước khi dùng số liệu benchmark cũ cho triển khai mới.

API kiểm tra điều kiện gate, deduplicate bằng khóa job/ảnh; request lặp trả cùng ID, không enqueue lại. `resident_queue/*.event.json` lưu track, số quan sát, vị trí tâm, ROI overlap và thời điểm gửi; file kết quả lưu xác suất và thời gian TensorRT/copy. UI đo riêng thời gian sự kiện→nhận kết quả. Nếu không ghép được track hoặc AI lỗi, không gán nhãn tốt cho sản phẩm chưa kiểm tra.

Heartbeat dùng publish atomic có retry khi Windows/DrvFS giữ read handle; lỗi ghi telemetry ngắn không được làm worker unload engine. Kiểm thử giữ khóa đọc file trạng thái 2,3 giây đã xác nhận worker vẫn cùng generation và ready sau khi nhả khóa. API cũng đã từ chối sự kiện chưa cắt ROI và trả lại cùng request ID cho sự kiện cũ mà không tăng lượt GPU; bằng chứng `artifacts/b1_stream/roi_api_smoke.json`.

Đã chạy `node tests/test_conveyor_gate.cjs`: nền trống/nhiễu, thay đổi sáng toàn cảnh, không gọi trước ROI, một lần/vật thể, sản phẩm kế tiếp và vật thể đứng tại ROI đều qua. Kiểm thử bằng thao tác file chooser trên browser: 3 ảnh tự vào băng chuyền, 3 track, 3 sự kiện, 3 kết quả; mỗi sự kiện có 69–70 quan sát, overlap=1.0; sau khi ra hết không có sự kiện thêm. Đây là kiểm tra chức năng mô phỏng, chưa chứng minh gate recall hay chất lượng trong camera/ánh sáng/vật thể thực.

## Mục tiêu

Giảm số lần gọi TensorRT bằng cách dùng xử lý ảnh nhẹ để trả lời hai câu hỏi: có sản phẩm trong vùng kiểm tra không, và frame nào đại diện tốt nhất cho sản phẩm đó? Cổng này không thay CNN quyết định lỗi. Nó chỉ chọn thời điểm và vùng không gian cần kiểm tra; CNN vẫn phân loại `good / bent / loose / missing / pierced`.

## Giữ đầu vào CNN đúng với mô hình đã huấn luyện

B1 triển khai ở đây nhận toàn ảnh sản phẩm resize 384×384; `nvstreammux` trong luồng RTSP cũ dùng bề mặt trung gian 640×640. Nó học cả sản phẩm, ống hút và ngữ cảnh ảnh; chưa được huấn luyện với crop ROI của cảnh băng chuyền. Vì vậy ROI dùng để phát hiện sản phẩm và kích hoạt suy luận. Khi cổng mở, gửi **ảnh sản phẩm gốc đầy đủ** vào TensorRT với đúng resize/chuẩn hóa. Không đưa cảnh ghép hoặc crop tùy ý vào B1 rồi dùng kết quả B0–B6 để khẳng định chất lượng trên đầu vào mới.

ROI được hiệu chuẩn theo camera, lưu dưới dạng đa giác/tỉ lệ tọa độ để không phụ thuộc độ phân giải. Có thể có một vùng vào, một vùng kiểm tra và một đường kích hoạt. Nếu camera lắp cố định, ROI loại phần nền/cạnh băng chuyền, nhưng phải chứa trọn sản phẩm cùng ống hút và vùng lân cận cần phân biệt lỗi.

## Logic truyền thống đề xuất

Ở nhánh điều khiển, giảm frame xuống khoảng 160×90 hoặc 320×180 và xử lý luma trong ROI. Không cần chạy CNN trên frame này. Từ ảnh nền đã hiệu chuẩn `B_t`, tính sai khác `D_t(x,y)=|I_t(x,y)-B_t(x,y)|`, ngưỡng hóa thành mặt nạ tiền cảnh `M_t=[D_t>τ]`, rồi dùng opening/closing để bỏ nhiễu nhỏ và nối vùng sản phẩm. Nền chỉ được cập nhật ở vùng không bị mặt nạ che để sản phẩm đứng yên không bị hấp thụ vào nền. Ánh sáng thay đổi toàn khung cần bù sáng hoặc chuyển sang so sánh gradient/độ bão hòa.

Với thành phần liên thông lớn nhất trong ROI, theo dõi diện tích tương đối `a_t=|M_t|/|ROI|`, tâm `c_t`, tỉ lệ khung và độ tin cậy của mask. Chỉ coi là sản phẩm khi các điều kiện diện tích/kích thước nằm trong khoảng đã hiệu chuẩn và tồn tại ổn định trong `N` trên `M` frame. Một đường ảo ở vùng kiểm tra giúp kích hoạt một lần khi tâm sản phẩm cắt đường. Không dùng ngưỡng diện tích đơn lẻ như quyết định cuối: bóng, phản quang hoặc biến động sáng có thể tạo mask giả.

Đầu cổng là một máy trạng thái theo từng vật thể:

```text
IDLE → CANDIDATE → CONFIRMED → INFER_ONCE → EXIT/COOLDOWN → IDLE
```

`IDLE`: chỉ lấy mẫu rẻ để tìm tiền cảnh. `CANDIDATE`: kiểm tra hình học và độ bền qua nhiều frame. `CONFIRMED`: vật thể đã vào ROI kiểm tra hoặc cắt đường kích hoạt. `INFER_ONCE`: chọn frame rõ nhất trong cửa sổ ngắn rồi gọi B1 một lần. `EXIT/COOLDOWN`: giữ kết quả gắn với sản phẩm và không gọi lặp cho đến khi vùng kiểm tra trống đủ lâu. Như vậy cooldown gắn với sự kiện vào/ra, không phải bộ đếm thời gian tùy ý làm bỏ qua sản phẩm kế tiếp.

Trong vài frame ứng viên, chọn frame có độ nét cao (phương sai Laplacian trên ảnh giảm kích thước), ít nhòe và sản phẩm nằm trọn trong ROI. Nếu sản phẩm đứng yên, có thể đợi đến khi chuyển động thấp; nếu sản phẩm đang chạy liên tục, kích hoạt theo đường cắt thay vì đợi đứng yên. Một giới hạn tần suất suy luận vẫn dùng làm bảo vệ tải, nhưng không thay thế phát hiện vào/ra.

## Quy tắc an toàn khi cổng không chắc chắn

Cổng tối ưu recall, không tối ưu số frame bỏ. Nếu mask không rõ, sản phẩm chạm biên ROI, diện tích bất thường, nhiều vùng dính nhau, ánh sáng thay đổi mạnh hoặc quá thời hạn mà chưa xác nhận được, chuyển sang `UNCERTAIN` và gửi frame gốc vào CNN (fail-open). Không có sản phẩm thì không gọi CNN. Sau mỗi lần gọi, lưu class, xác suất, thời gian, ROI/mask và lý do kích hoạt; kết quả giữ lại trên UI chỉ gắn với đúng sự kiện/vật thể.

Nếu nhiều sản phẩm chồng lấp hoặc đi sát nhau khiến mask không tách được, bộ phân loại một ảnh không đủ để định danh từng sản phẩm. Khi đó cần detector/segmenter và tracker, hoặc cơ cấu cách ly vật thể trên băng chuyền. Không được hiển thị một nhãn mới nhất như thể nó chắc chắn thuộc về mọi sản phẩm trên màn hình.

## Vị trí trong DeepStream

Pipeline dự kiến:

```text
RTSP → NVDEC/NVMM → nvstreammux
     → cổng ROI/foreground và chọn frame (trước nvinfer)
     → chỉ buffer được chọn → nvinfer TensorRT B1 → kết quả theo PTS/sự kiện
     ↘ nhánh preview nhận mọi frame, vẽ ROI/trạng thái cổng
```

Probe/cổng phải giữ nguyên buffer NVMM gốc khi chuyển frame đã chọn vào `nvinfer`; chỉ phép đo luma thấp phân giải dùng nhánh phụ. Preview tách khỏi nhánh suy luận để frame bị bỏ qua vẫn hiện liên tục. Dùng PTS để ghép sự kiện và kết quả; khi không ghép được thì báo “kết quả mới nhất” cùng độ trễ, không gán nhầm nhãn cho frame/vật thể khác. Bản hiện hành vẫn suy luận mọi frame; cổng này là thiết kế phiên bản tiếp theo, chưa được bật trong demo TensorRT đang chạy.

## Phép thử cần có trước khi kết luận

Tập ảnh phân loại hiện tại không có video băng chuyền, ROI/box hay nhãn thời điểm vật thể vào-ra. Nó đủ đánh giá CNN trên ảnh, nhưng **không đủ để chứng minh cổng ROI không bỏ sót sản phẩm**. Clip replay hiện tại chuyển sample qua từng frame nên cũng không tương đương một vật thể đi qua camera theo thời gian. Không dùng accuracy validation hiện tại để tuyên bố cổng đã được kiểm chứng.

Tạo bộ video sự kiện từ ảnh có nhãn: đặt từng sản phẩm vào nền/camera cố định, quay đoạn vào–qua ROI–ra ở nhiều tốc độ, vị trí, ánh sáng, blur và khoảng cách; gán ID sản phẩm, class lỗi, thời điểm/ROI vào-ra và frame CNN nên chọn. Tách video theo sản phẩm/phiên quay để tránh cùng một sản phẩm lọt vào cả tuning lẫn test. Hiệu chuẩn `τ`, diện tích, số frame debounce trên video validation; khóa cấu hình trước khi chạy test.

So sánh ít nhất hai hệ thống trên cùng video: (A) B1 gọi mọi frame; (B) gate + B1. Báo cáo event recall của gate, số lần gọi AI trên mỗi sản phẩm, tỷ lệ sản phẩm không được suy luận, accuracy/macro-F1/FNR theo sản phẩm sau khi gate, false trigger/phút, latency từ lúc vào ROI đến kết quả, GPU utilization và CPU/memory của gate. Báo cáo thêm p50/p95 và chế độ fail-open. Tiêu chí đầu tiên là không tăng bỏ sót lỗi quá mức đã định trước; chỉ sau đó mới tối ưu số lần suy luận.

## Lộ trình triển khai

1. Ghi video ngắn từ camera cố định và hiệu chuẩn ROI bằng giao diện kéo thả; bắt đầu ở 5–10 lần kiểm tra mask mỗi giây.
2. Chạy gate ở chế độ **shadow**: ghi các frame nó sẽ chọn nhưng vẫn gọi CNN mọi frame. Đo gate recall và căn PTS, chưa điều khiển inference.
3. Chạy trên clip đã gán sự kiện; đặt ngưỡng fail-open để không chắc chắn thì vẫn gọi CNN.
4. Bật drop trước `nvinfer`, tách nhánh preview để xem mọi frame và đánh giá lại chất lượng/số lần gọi/độ trễ.
5. Chỉ cân nhắc crop ROI, tracking nhiều sản phẩm hoặc TensorRT cho gate sau khi thu thập đủ video và huấn luyện/đánh giá lại pipeline tương ứng.
