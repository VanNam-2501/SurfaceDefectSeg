# AI Development Log

## Cách dùng và giới hạn của bản hiện tại

Đây là bản hồi cứu từ các task quan trọng còn thấy trong lịch sử làm việc và
các file đã tạo. Trước khi nộp, sinh viên phải:

1. thay `CẦN XÁC NHẬN` bằng exact tool version/model ID hiển thị trong ứng dụng;
2. export raw conversation/task vào `PROMPT_ARCHIVE/`;
3. điền commit thật sau khi repository chính được thiết lập;
4. bổ sung mọi công cụ AI khác đã dùng ngoài Codex;
5. kiểm tra lại mô tả “chấp nhận/chỉnh sửa/loại bỏ” bằng hiểu biết cá nhân.

Các đường dẫn trong phiên trước 2026-08-19 có thể là đường dẫn lịch sử. Dùng
`../../docs/MIGRATION_MANIFEST.md` để ánh xạ sang cấu trúc hiện hành.

AI không chịu trách nhiệm thay sinh viên về tính chính xác, bản quyền, an toàn
hoặc khả năng vận hành.

---

## AI-2026-08-15-01 — Công cụ review và sửa dữ liệu

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN exact model/build**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-15. Xây dựng công cụ review nhanh cho
dataset segmentation có nghi ngờ label sai, false positive, mask lỗi và cần làm
việc với nhiều ảnh.

**3. Prompt gốc:**

> “thế giờ bạn hãy tạo cho tôi 1 công cụ thật tốt để tôi có thể xử lý lại data,
> có thể xử lý được hết những vấn đề trên thật thuận tiện làm nhanh và làm nhiều data được”

**Prompt hiệu chỉnh tiêu biểu:**

> “muốn xem của toàn bộ dữ liệu luôn”

> “cho thêm 1 option là lấy mask của model dự báo làm mask”

> “thiết kế như nào đó, với những ảnh tôi k review và ảnh đã review khi xuất thì
> có sẵn bộ data đường dẫn các thứ phù hợp cả ảnh gốc lẫn mask để train, val, test”

**4. File/thành phần liên quan:** `dataset_review_tool/`, dataset audit manifests,
prediction export scripts và mask review UI.

**5. Kết quả AI:** Công cụ web local để lọc mẫu nghi vấn, xem ảnh/GT/prediction,
sửa hoặc lấy prediction làm mask, lưu tiến độ review và xuất dataset.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận cơ chế lưu tiến độ và review queue;
chỉnh UI nhiều lần để xem ảnh đồng thời; bổ sung lựa chọn lấy prediction mask;
loại bỏ cách trình bày buộc người dùng mở từng ô.

**7. Lý do chỉnh sửa:** Tăng tốc review hàng nghìn ảnh, tránh bỏ mất ảnh gốc/GT,
và làm rõ “mẫu nghi vấn” không đồng nghĩa label chắc chắn sai.

**8. Kiểm thử/xác minh:** Kiểm tra thủ công qua ảnh chụp màn hình, thử sửa/xóa
mask, kiểm tra lưu tiến độ và đường dẫn export. **Cần bổ sung test tự động và biên
bản export dataset cụ thể trước khi nộp.**

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-16-02 — Pipeline Kaggle và VMamba

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-16 đến 2026-08-18. Chuẩn bị Kaggle
runtime, wheel selective scan, chạy riêng VMamba và đánh giá đủ ba model.

**3. Prompt gốc:**

> “setup ddeer toi train tren moi thu tren kaggle ay nhu cu”

**Prompt hiệu chỉnh tiêu biểu:**

> “train từng cái đi chứ k train 3 cái 1 lần thử train val test thí nghiệm trên mamba trước”

> “GPU Memory 1.4GiB Max 15GiB chỉ dùng từng này vram phí quá tăng bz đi”

> “train rồi val rồi test rồi thí nghiệm hết luôn nha”

**4. File/thành phần liên quan:** `train_on_kaggle.py`, `train_common.py`,
`setup_vmamba_kaggle.sh`, `TTTN_All_Models_Kaggle_Full_Pipeline.ipynb`,
`kaggle_upload/`, `vmamba_t.py`.

**5. Kết quả AI:** Sửa CLI training, chuẩn bị notebook/ZIP Kaggle, cấu hình
VMamba wheel và các lệnh Validation/Test.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận chạy từng model; tăng batch theo
VRAM thực tế; bỏ kế hoạch train ba model đồng thời; về sau chỉ dùng checkpoint
có sẵn để đánh giá.

**7. Lý do chỉnh sửa:** Giới hạn GPU/Kaggle, thời gian VMamba và yêu cầu tận dụng
VRAM. Cấu hình thực tế cuối khác protocol batch ban đầu và đã được ghi tại
`FINAL_EXPERIMENT_CONFIG.md`.

**8. Kiểm thử/xác minh:** VMamba checkpoint đạt best epoch 23, U-Net 48,
SegFormer 50; checkpoint chứa config/model/optimizer và tải được local. Split
hash được `check_protocol.py` xác nhận PASS.

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-16-03 — Logic giảm báo động giả

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-16 đến 2026-08-17. Giảm FPR nhưng vẫn
bảo vệ defect nhỏ; threshold chỉ chọn trên Validation.

**3. Prompt gốc:**

> “sau khi xử lý lại data thì cái phần logic quyết định thì làm sao để cân bằng
> giữa việc bỏ sót bất thường và báo động giả”

**Prompt hiệu chỉnh tiêu biểu:**

> “với những bất thường nhỏ thì có ổn k”

> “không có cái review á logic nãy bạn đã chốt rồi mà”

**4. File/thành phần liên quan:** `decision_policy.py`,
`adaptive_component_policy.py`, `calibrate_decision_policy.py`,
`evaluate_decision_policy.py`, `run_all_decision_experiments.py`.

**5. Kết quả AI:** ROI hợp lệ, connected component, confidence/area/persistence,
FNR-constrained selection và fully automatic Good/Defect comparison.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận component evidence và Validation
constraint; ban đầu có PASS/REVIEW/DEFECT nhưng scope kết quả cuối chuyển sang
fully automatic Good/Defect.

**7. Lý do chỉnh sửa:** Review không phải kết luận tự động; yêu cầu cuối cần
so sánh trực tiếp FPR/FNR và triển khai tự động.

**8. Kiểm thử/xác minh:** `test_decision_policy.py` PASS 3/3; adaptive scan có
10.720 cấu hình/model; policy khóa trên Validation rồi báo cáo Test.

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-17-04 — Learned verifier và hybrid

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-17 đến 2026-08-18. Thử mô hình quyết
định nhỏ và kết hợp từng cặp U-Net, SegFormer, VMamba.

**3. Prompt gốc:**

> “có cách nào model riêng mà cân bằng giảm báo động giả hơn nữa k”

**Prompt hiệu chỉnh tiêu biểu:**

> “thử đi”

> “giờ có đủ model rồi chạy rồi cho tôi xem bảng kết quả”

**4. File/thành phần liên quan:** `learned_decision_verifier.py`,
`test_learned_decision_verifier.py`, `run_all_decision_experiments.py`,
`final_thesis_deliverables/decision_and_test_audit/learned_all3/` và
`hybrid_pairs/`.

**5. Kết quả AI:** HistGradientBoosting verifier, 5-fold OOF Validation,
top-k score selector, fusion features và hybrid specialist rescue.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận HGB cho U-Net và fusion; selector
loại HGB cho SegFormer/VMamba vì `top_4_mean` Pareto tốt hơn trên Validation;
hybrid U-Net+VMamba được giữ làm ứng viên triển khai.

**7. Lý do chỉnh sửa:** Không ép model learned nếu statistic đơn giản tốt hơn;
tránh dùng Test để chọn chiến lược.

**8. Kiểm thử/xác minh:** Test verifier PASS 3/3. Hybrid U-Net+VMamba trên Test:
FPR 10,22%, FNR 3,81%; learned fusion ba model: FPR 12,69%, FNR 3,30%.

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-18-05 — Tổng hợp thí nghiệm và audit toàn bộ dữ liệu

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-18. Chạy Validation/Test đúng protocol,
tổng hợp E0/E2–E5/E7/E8, decision experiments và kiểm tra mọi ảnh để tìm label noise.

**3. Prompt gốc:**

> “tôi muốn chạy tất cả thí nghiệm á vì giờ có đủ model rồi chạy rồi cho tôi xem bảng kết quả”

**Prompt hiệu chỉnh tiêu biểu:**

> “E1 k cần E6 tôi bổ sung cái train của mamba sau, k cần e9 e10 e11”

> “còn test trên toàn bộ ảnh để xem nó sai ở đâu là tôi muốn kiểm tra lại data ấy,
> xem những case nào đang sao, do label hay do model”

**4. File/thành phần liên quan:** `compile_thesis_evaluation_report.py`,
`audit_full_dataset_labels.py`, `build_test_case_audit.py`,
`run_all_decision_experiments.py`, `final_thesis_deliverables/`.

**5. Kết quả AI:** Ba bộ deliverable: thesis evaluation, decision/test audit và
full-dataset audit cho 7.168 ảnh.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận scope E0/E2–E5/E7/E8; loại
E1/E6/E9/E10/E11 khỏi lần báo cáo hiện tại; giữ data audit tách khỏi model selection.

**7. Lý do chỉnh sửa:** Không có đủ bằng chứng công bằng cho E6 và không muốn
phình scope; tránh dùng toàn bộ data audit để tune theo Test.

**8. Kiểm thử/xác minh:** Split/content/file checks PASS; mask integrity issues
CSV có 0 dòng; mọi threshold/policy có metadata Validation selection.

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-18-06 — Trực quan hóa kết quả

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-18. Tạo biểu đồ đầy đủ cho báo cáo.

**3. Prompt gốc:**

> “final_thesis_deliverables đã có tất cả kết quả ở đây tạo biểu đồ trực quan
> đầy đủ cho các thí nghiệm kết quả của tôi”

**4. File/thành phần liên quan:** `generate_thesis_visualizations.py`,
`final_thesis_deliverables/visualizations/`.

**5. Kết quả AI:** 20 PNG, 19 SVG, dashboard HTML, chart manifest và bảng tóm tắt.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận màu nhất quán và dashboard; chỉnh
bar chart về baseline 0, sửa nhãn scatter chồng nhau và contact sheet.

**7. Lý do chỉnh sửa:** Tránh biểu đồ gây hiểu sai, tăng khả năng đọc và dùng
được trực tiếp trong luận văn.

**8. Kiểm thử/xác minh:** Pillow verify toàn bộ 20 PNG; kiểm tra số SVG/manifest;
QA trực quan các hình E2, FPR–FNR, audit và qualitative gallery.

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-19-07 — Cleanup và hồ sơ tuân thủ

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 family — **CẦN XÁC NHẬN**.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-19. Làm sạch dự án, audit sản phẩm phải
nộp và tạo hồ sơ AI ban đầu.

**3. Prompt gốc:**

> “clean lại dự án và xem tôi còn thiếu những yêu cầu nào”

**4. File/thành phần liên quan:** root README/.gitignore,
`FINAL_EXPERIMENT_CONFIG.md`, `verify_submission.ps1`, `submission/`,
`start_decision_demo.ps1`, tài liệu demo và unit test verifier.

**5. Kết quả AI:** Checklist bằng chứng, skeleton hồ sơ nộp, AI log hồi cứu,
test report, checkpoint/archive inventory và demo dùng đường dẫn local.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chỉ xóa cache/log tái tạo được; không xóa
dataset, checkpoint, ZIP Kaggle hoặc kết quả; không giả tạo Git history.

**7. Lý do chỉnh sửa:** Bảo toàn bằng chứng và đường dẫn đang chạy trong khi
làm rõ thiếu sót thực tế.

**8. Kiểm thử/xác minh:** Protocol PASS; CPU tests 10/10; web build/test 2/2;
demo policy/checkpoint path được kiểm tra tồn tại.

**9. Commit:** `PENDING_NO_ROOT_GIT`.

---

## AI-2026-08-19-08 — Tái cấu trúc repository và xác minh bàn giao

**1. Công cụ/model:** OpenAI Codex desktop, GPT-5 — cần bổ sung exact
deployment/build từ task export nếu UI cung cấp.

**2. Ngày, mục tiêu, ngữ cảnh:** 2026-08-19. Thiết kế lại tên/cấu trúc dự án,
giữ an toàn dữ liệu và thiết lập một repository nguồn duy nhất.

**3. Prompt gốc:**

> “thiết kế lại dự án từ cách đặt tên đến cấu trúc theo đúng yêu cầu chưa”

**Prompt hiệu chỉnh:**

> “thực hiện đi”

**4. File/thành phần liên quan:** toàn bộ cây `apps/`, `src/`, `scripts/`,
`tests/`, `docs/`, `experiments/`, ba launcher gốc, `.gitignore`, `submission/`.

**5. Kết quả AI:** Di chuyển sang cây chuẩn, sửa đường dẫn tương đối, cập nhật
notebook/tài liệu, khởi tạo Git root, giữ dữ liệu lớn ngoài Git và lập hồ sơ
migration có kiểm tra hash.

**6. Chấp nhận/chỉnh sửa/loại bỏ:** Chấp nhận kiến trúc một repo; giữ mã VMamba
vendor nhưng loại metadata Git lồng; giữ toàn bộ dữ liệu cũ trong `archive`.

**7. Lý do chỉnh sửa:** Tách mã nguồn khỏi dữ liệu/kết quả, tránh đường dẫn cứng,
giúp test, bàn giao và giải trình mã dễ hơn mà không mất bằng chứng.

**8. Kiểm thử/xác minh:** SHA-256 cho nguồn/report khi copy; PowerShell parse,
Python compile, notebook JSON, protocol, 15 test Python, web build + 2 test và
FastAPI path check đều PASS.

**9. Commit:** `PENDING_BASELINE_COMMIT`.
