# Bảng Kết Quả Đánh Giá Trên Tập Kiểm Thử Khóa (Locked Test Set)

| architecture                  | condition        |    fnr | fnr_ci_95        |    fpr |   recall_good |   recall_bent |   recall_loose |   recall_missing |   recall_pierced |   macro_recall_defect |   review_rate |   reweighted_acc_deploy |   cost_per_1000_cartons |   latency_ms |
|:------------------------------|:-----------------|-------:|:-----------------|-------:|--------------:|--------------:|---------------:|-----------------:|-----------------:|----------------------:|--------------:|------------------------:|------------------------:|-------------:|
| Two-Stage Cascade             | Clean            | 0      | [0.0000, 0.0158] | 0      |        0      |        0      |         0.3167 |           0.3833 |           0.0833 |                0.1958 |        0.83   |                  0.0098 |                 9893.75 |         5.67 |
| Two-Stage Cascade             | Factory Degraded | 0      | [0.0000, 0.0158] | 0      |        0      |        0      |         0.05   |           0.0833 |           0      |                0.0333 |        0.9733 |                  0.0017 |                 9983.33 |         5.75 |
| Single-Stage Baseline (full5) | Clean            | 0.05   | [0.0288, 0.0853] | 0.3    |        0.7    |        0.6833 |         0.95   |           1      |           0.65   |                0.8208 |        0      |                  0.706  |                  535    |         4.9  |
| Single-Stage Baseline (full5) | Factory Degraded | 0.0583 | [0.0351, 0.0955] | 0.4167 |        0.5833 |        0.65   |         0.6167 |           1      |           0.8    |                0.7667 |        0      |                  0.5925 |                  687.5  |         4.8  |

*Ghi chú: Đánh giá duy nhất một lần (single-pass locked test). Chi phí dự tính trên 1.000 hộp đồ uống dựa trên ma trận chi phí c_FN:c_REVIEW:c_FP = 100:10:1 và tỉ lệ lỗi sản xuất pi_deploy = 5%.*
