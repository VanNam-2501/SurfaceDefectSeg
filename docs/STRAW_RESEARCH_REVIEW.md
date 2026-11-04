# Đánh giá lại độ sâu nghiên cứu — 2026-10-18

> **Đọc kế hoạch thống nhất mới:** [STRAW_RESEARCH_PLAN.md](STRAW_RESEARCH_PLAN.md).
> Nội dung dưới đây lưu đánh giá của các hướng trước; nghiên cứu hiện tại tập
> trung phân loại 5 trạng thái bằng CNN. LOBO và REVIEW không thuộc lõi.

Phạm vi theo xác nhận mới nhất: chỉ bộ dữ liệu ống hút hiện có. Định hướng
model/training cụ thể và cách đánh giá tổng quát giữa thương hiệu nằm tại
[INSPECTION_MODEL_RESEARCH.md](INSPECTION_MODEL_RESEARCH.md). Không yêu cầu
thêm dataset sản phẩm khác để thực hiện khóa luận trong phạm vi đã chốt.

## Kết luận hiện tại

Pipeline đã được kiểm tra kỹ thuật nhưng **chưa có bằng chứng để gọi phương
pháp là tốt hoặc khóa luận là mạnh**. 72 là số lượt huấn luyện theo một kế
hoạch dự kiến, không phải 72 đóng góp hay 72 thí nghiệm khoa học độc lập.
Chưa nên dùng toàn bộ ngân sách GPU trước khi kiểm chứng giả thuyết chính.

## Các điểm yếu cụ thể

1. Hai tầng, temperature scaling và confidence/margin gate đã có trong thiết kế;
   chưa chứng minh hệ thống hai tầng đem lại lợi ích so với full5 chọn lọc.
   Cơ chế từ chối dự đoán là một hướng đã có nghiên cứu, không tự là tính mới.
2. Đối chứng hiện tại chưa đủ cân xứng: full5 chọn lọc dùng một ngưỡng xác suất
   lớn nhất, còn cascade có hiệu chỉnh prior và nhiều gate. Cải thiện, nếu có,
   chưa thể quy hết cho kiến trúc hai tầng.
3. Resize toàn ảnh xuống 224/384 có thể làm mất chi tiết lỗi nhỏ. Đây là giả
   thuyết cần kiểm tra bằng phân tích ảnh/lỗi, chưa phải nguyên nhân đã xác nhận.
4. Nhóm 76 cặp dHash là biện pháp phòng rò rỉ bảo thủ, chưa thay thế việc rà nhãn
   và xác minh quan hệ giữa các ảnh/phiên chụp.
5. 1% FNR, 30% deployment REVIEW, prior 5% và ma trận chi phí là giả định.
   Chưa có yêu cầu vận hành hoặc số liệu doanh nghiệp để gọi chúng là chuẩn công nghiệp.
6. Toàn bộ bộ dữ liệu đã có lịch sử dùng để phát triển. LOBO nội bộ hữu ích,
   nhưng chưa chứng minh kết quả ở nhà máy hay trên bộ ảnh ngoài độc lập.
7. `make_straw_masks.py` sinh pseudo-mask theo `class_name`, kể cả vùng kỳ vọng
   riêng cho Missing. Không được coi các mask này là ground truth định vị độc
   lập hoặc dùng nhãn thật để chọn ROI khi suy luận. Pipeline mới hiện không
   dùng các mask này cho augmentation safe.
8. Chưa có training run đầy đủ mới; smoke chỉ kiểm tra mã. Demo cũ chưa dùng
   policy mới và các mô tả causal/predictive maintenance chưa có kiểm chứng tương ứng.

## Đối chiếu tài liệu

- *Deep Learning-Based Visual Inspection of Straw Defects in Beverage Box
  Packaging for Process Quality Monitoring*, Bem Gumapac, trang abstract/poster
  hội nghị Sciforum, ngày công bố trên trang 15-10-2026:
  https://sciforum.net/paper/34988
  Abstract mô tả cùng dataset 2.000 ảnh, DenseNet121/Xception/InceptionV3,
  fine-tuning và Grad-CAM. Đây là bằng chứng đã có bài toán/đối chứng tương tự;
  không so trực tiếp số accuracy với dự án khi split và protocol chưa được đối chiếu.
- *Selective Classification for Deep Neural Networks*, Geifman & El-Yaniv:
  https://arxiv.org/abs/1705.08500
  Cơ chế từ chối gắn với đánh đổi risk–coverage đã có tiền lệ.
- *Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming*, ICML 2022:
  https://proceedings.mlr.press/v162/wen22d.html
  Động cơ kiểm tra mô hình dựa vào đặc trưng liên quan/nền có cơ sở nghiên cứu;
  chưa có bằng chứng rằng dự án cụ thể đang mắc shortcut này.

## Hướng nghiên cứu nên ưu tiên kiểm chứng

**Giả thuyết:** kết hợp ngữ cảnh toàn hộp với vùng gắn ống hút giúp giữ chi tiết
lỗi và tổng quát giữa thương hiệu tốt hơn chỉ dùng toàn ảnh; cơ chế REVIEW
cần được so sánh ở cùng ngân sách kiểm tra thủ công.

Đây là giả thuyết ứng dụng, chưa được chứng minh và chưa xác nhận tính mới sau
khi rà đầy đủ tài liệu. Nếu không có cải thiện, báo cáo kết quả âm và phân tích
nguyên nhân; không thêm module chỉ để kéo dài tên phương pháp.

### Các so sánh tối thiểu trước khi mở rộng kiến trúc

| So sánh | Câu hỏi trả lời |
|---|---|
| Toàn ảnh → full5 | Baseline đủ mạnh sau fine-tuning đạt đến đâu? |
| Vùng gắn ống hút → full5 | Chi tiết vùng liên quan có ích không? |
| Toàn ảnh + vùng gắn ống hút → full5 | Ngữ cảnh và chi tiết có bổ sung nhau không? |
| Hai tầng với cùng kiểu đầu vào | Phân cấp có đem lại lợi ích riêng không? |
| Các phương án với cùng ngân sách REVIEW | Lợi ích còn tồn tại khi không cho một bên từ chối nhiều hơn? |
| Giữ riêng thương hiệu, kiểm tra vùng bao bì | Có dấu hiệu phụ thuộc đặc trưng thương hiệu không? |

Định vị ROI phải chạy được mà không biết nhãn lỗi: annotation vùng gắn ống hút
theo quy tắc chung hoặc một bộ định vị học chỉ từ Train. Với Missing, ROI là
vị trí gắn kỳ vọng; không dùng điều kiện `if ground_truth == Missing` tại inference.
Nếu dùng ROI được vẽ tay ở Test, phải ghi là thí nghiệm oracle để đo trần hiệu
năng, không coi đó là hệ thống tự động. Tách lỗi định vị khỏi lỗi phân loại và
đánh giá độ nhạy khi ROI lệch vị trí.

Ngưỡng ứng với ngân sách REVIEW được chọn bằng Validation rồi khóa; Test báo
cáo tỷ lệ REVIEW thực tế vì có thể khác Validation. Có thể vẽ đường risk–coverage
trên Test để mô tả, nhưng không dùng nó để chọn operating point triển khai.
Ưu tiên các ngân sách giả định 0%, 5%, 10%, 20% và công khai chúng là giả định.

### Thứ tự công việc

1. Rà ảnh/lỗi và nhãn; chốt định nghĩa ROI; lập bảng related work và khác biệt
   với bài cùng dataset. Không cần tăng backbone ở bước này.
2. Train một baseline pretrained đầy đủ; phân tích nhầm lớp và khác biệt thương
   hiệu bằng Validation, không chỉ xem accuracy tổng.
3. Pilot toàn ảnh/ROI/kết hợp trên một thiết lập giữ riêng thương hiệu, dùng
   phần dữ liệu phát triển; chưa xem bộ ảnh xác nhận độc lập tương lai.
4. Chỉ mở rộng 3 seed, các fold và ablation sau khi pilot trả lời được câu hỏi
   về biểu diễn đầu vào. Đăng ký lại plan và output riêng; không trộn với plan cũ.
5. Thu tập ảnh ngoài nếu có điều kiện; tích hợp đúng model/policy đã khóa vào
   demo, đo pipeline thực và phân tích thất bại trước khi viết kết luận ứng dụng.

Phần pilot ROI/kết hợp và đối chứng REVIEW cân xứng **chưa được triển khai**
trong pipeline hiện tại. Bản đánh giá này điều chỉnh ưu tiên nghiên cứu; không
phải thông báo rằng các cải tiến đã có kết quả.
