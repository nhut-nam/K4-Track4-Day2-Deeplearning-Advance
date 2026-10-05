# Báo Cáo Kết Quả Thực Nghiệm - DeepWeeds (Lab Day 2)

## 1. Lựa chọn Backbone (Bước 1)
Sau khi thử nghiệm qua nhiều kiến trúc khác nhau (ResNet, ConvNeXt, Swin, EfficientNet, MobileNet), mô hình **ResNet50** được chọn làm Baseline (`T00`).
- **Lý do**: ResNet50 mang lại sự cân bằng hoàn hảo giữa tốc độ huấn luyện, độ trễ suy luận (latency) và khả năng trích xuất đặc trưng ổn định trên tập dữ liệu cỏ dại (DeepWeeds).

## 2. Phân tích Công thức Huấn luyện (Bước 2 & 4)
Chúng tôi đã tiến hành so sánh cấu hình Mốc (Baseline `T00` - ResNet50 với CrossEntropy) và cấu hình Chung kết (`F01` - kết hợp Focal Loss và CutMix) thông qua 3 seed khác nhau (0, 1, 2) trên tập Test.

**Kết quả thu được:**
- **Baseline (T00)**: Top-1 Accuracy đạt **86.83%**, Macro-F1 đạt **0.8220**.
- **Final (F01)**: Top-1 Accuracy đạt **84.40%**, Macro-F1 đạt **0.7986**.

**Nhận xét:**
- Cấu hình phức tạp `F01` với các phương pháp Data Augmentation mạnh (CutMix) và cân bằng class (Focal Loss) thực tế lại **làm giảm hiệu suất** của mô hình.
- Hai lớp khó học là *Chinee apple* và *Snake weed* có Recall rất thấp (khoảng 52% và 75%).
- Lỗi ECE của `F01` (0.2445) cao hơn rất nhiều so với `T00` (0.0187), cho thấy mô hình bị nhiễu do CutMix.
=> **Kết luận**: Tập dữ liệu DeepWeeds có bối cảnh tự nhiên phức tạp, việc áp dụng CutMix trộn lẫn các bối cảnh làm mất đi thông tin không gian cục bộ của thực vật, dẫn đến phản tác dụng.

## 3. Tối ưu Suy luận và Độ trễ (Bước 3)
Chúng tôi đã tiến hành đo đạc độ trễ trên Google Colab (GPU Tesla T4, Batch Size = 1) và áp dụng một số kỹ thuật Test-Time.

- **TTA (Test-Time Augmentation)**: Lật ngang ảnh (Horizontal Flip) và gộp xác suất giúp điểm Val Macro-F1 tăng vọt từ **0.8146** lên **0.8314** (+0.0168). Điều này cho thấy tập đánh giá có nhiều góc độ phức tạp, và TTA giúp mô hình chốt dự đoán tự tin hơn.
- **Gộp BatchNorm (BN Fusion)**: Khi gộp BN vào Conv2d, thời gian trung bình (p50) không đổi (6.54ms), nhưng thời gian trễ tối đa (p95) giảm mạnh từ **11.72ms xuống chỉ còn 6.98ms**, giúp hệ thống chạy ổn định và mượt mà hơn rất nhiều, rất phù hợp khi triển khai trên robot thực tế.
- **Mixed Precision (AMP)**: Do batch_size = 1, chi phí overhead khi ép kiểu dữ liệu khiến độ trễ p50 tăng nhẹ lên **7.79ms**. AMP sẽ không phù hợp cho suy luận từng ảnh một.
