# Bài Tập Lớn Cuối Kỳ Môn Học Máy (Machine Learning Final Project)

Repository này lưu trữ mã nguồn, dữ liệu và báo cáo kết quả thực hiện bài tập lớn cuối kỳ môn Học máy. Dự án bao gồm hai bài toán kinh điển trong Machine Learning: **Hồi quy tuyến tính (Linear Regression)** và **Phân loại nhị phân (Binary Classification)**.

## 📂 Cấu trúc Repository

- `BTL_HoiQuyTuyenTinh_final.ipynb`: Jupyter Notebook thực hiện phân tích, huấn luyện và đánh giá mô hình cho bài toán dự đoán giá nhà.
- `BTL_PhanLoaiNhiPhan_final.ipynb`: Jupyter Notebook thực hiện xử lý dữ liệu, tối ưu siêu tham số và huấn luyện mô hình cho bài toán phát hiện giao dịch gian lận.
- `USA_Housing.csv`: Bộ dữ liệu sử dụng cho bài toán hồi quy tuyến tính.
- `credit_card_fraud_synthetic.csv`: Bộ dữ liệu tổng hợp sử dụng cho bài toán phân loại nhị phân.

---

## 🚀 Chi tiết các bài toán

### 1. Dự đoán giá nhà (Hồi quy tuyến tính)
Bài toán tập trung xây dựng mô hình học máy thuộc nhóm Học có giám sát (Supervised Learning) nhằm dự đoán giá trị thực của một căn nhà (`Price`) dựa trên các đặc trưng kinh tế - xã hội và nhân khẩu học trung bình của khu vực xung quanh.

* **Phân tích Bộ dữ liệu (Exploratory Data Analysis - EDA):**
  * **Dataset:** `USA_Housing.csv`
  * **Biến mục tiêu (Target):** `Price` (Biến số thực liên tục).
  * **Đặc trưng đầu vào (Features):** Thu nhập trung bình (`Avg. Area Income`), Tuổi thọ trung bình của nhà (`Avg. Area House Age`), Số phòng/phòng ngủ trung bình (`Avg. Area Number of Rooms/Bedrooms`), và Dân số khu vực (`Area Population`).
  * **Đặc điểm dữ liệu:** Bộ dữ liệu có chất lượng lý tưởng (rất sạch), hoàn toàn không chứa giá trị khuyết thiếu (missing values). Qua phân tích tương quan, không phát hiện hiện tượng đa cộng tuyến (multicollinearity) nghiêm trọng giữa các biến độc lập. Hầu hết các đặc trưng (đặc biệt là Thu nhập trung bình) đều thể hiện mối quan hệ tuyến tính dương rất mạnh mẽ với biến mục tiêu `Price`.

* **Tiền xử lý dữ liệu (Data Preprocessing):**
  * Loại bỏ cột văn bản `Address` ra khỏi quá trình huấn luyện do định dạng text thô không thể đưa trực tiếp vào mô hình tuyến tính mà không qua xử lý NLP.
  * **Chuẩn hóa dữ liệu (Scaling):** Đặc biệt quan trọng khi áp dụng các mô hình có sử dụng kỹ thuật điều chuẩn (Regularization) như Ridge hay ElasticNet để đảm bảo các biến có cùng thang đo, tránh việc mô hình thiên vị các biến có giá trị tuyệt đối lớn.

* **Chiến lược Huấn luyện & Lựa chọn Mô hình (Modeling Strategy):**
  * **Thuật toán áp dụng:** Dự án tiến hành thực nghiệm và đối chiếu hiệu năng giữa 3 mô hình từ cơ bản đến nâng cao:
    1. **Linear Regression (Hồi quy tuyến tính đa biến):** Làm mô hình cơ sở (baseline).
    2. **Ridge Regression (L2 Regularization):** Xử lý hình phạt trên bình phương trọng số, giúp mô hình ổn định hơn.
    3. **ElasticNet Regression (L1 + L2 Regularization):** Kết hợp cả phương pháp triệt tiêu biến yếu và kiểm soát trọng số.
  * **Đánh giá chéo (Cross-Validation):** Tuyệt đối không chọn mô hình thông qua hiệu suất trên tập Test. Dự án sử dụng **K-Fold Cross-Validation** trên tập Train với tiêu chí `CV_RMSE_Mean` (Trung bình Lỗi bình phương trung bình căn). Điều này đảm bảo mô hình có khả năng tổng quát hóa tốt nhất trên dữ liệu chưa từng thấy, còn tập Test được cách ly hoàn toàn cho bước nghiệm thu cuối cùng.

* **Kết quả phân tích & Đánh giá:**
  * Theo tiêu chí `CV_RMSE_Mean`, **Ridge Regression** vươn lên là mô hình dẫn đầu với độ ổn định cao nhất. Sau khi xác định được Ridge là thuật toán tối ưu, mô hình mới được `fit` lại trên toàn bộ tập Train và đưa ra dự đoán trên tập Test.
  * Trong khi đó, **ElasticNet** cho kết quả tụt hậu. Điều này phản ánh việc tham số điều chuẩn áp dụng có thể quá mạnh, hoặc tính chất loại bỏ biến (L1) không phù hợp với bộ dữ liệu này (vì thực tế mọi biến số trong data đều đóng góp giá trị thông tin cao cho giá nhà).

* **Hạn chế & Hướng phát triển:** * Dataset hiện tại mang tính chất lý tưởng hóa và đơn giản. Thiếu vắng các yếu tố vi mô mang tính quyết định trong bất động sản thực tế như: diện tích mét vuông cụ thể, vị trí địa lý chính xác (tọa độ), tiện ích xung quanh, hay tình trạng xây dựng.
  * Cột `Address` đang bị lãng phí. Trong tương lai, có thể áp dụng các kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) để trích xuất từ khóa, tên bang/thành phố nhằm tạo ra các đặc trưng phân loại (categorical features) mới.
  * Các mô hình tuyến tính chỉ học được các mối quan hệ đường thẳng. Có thể thử nghiệm thêm các mô hình phi tuyến như Random Forest Regressor hoặc XGBoost để nắm bắt các cấu trúc dữ liệu phức tạp hơn.

---

### 2. Phát hiện gian lận thẻ tín dụng (Phân loại nhị phân)
Đây là bài toán trọng tâm đòi hỏi nhiều kỹ thuật xử lý dữ liệu phức tạp, nhằm nhận diện các giao dịch bất thường (Fraud) để bảo vệ rủi ro tài chính, trong bối cảnh dữ liệu mất cân bằng nghiêm trọng.

* **Bộ dữ liệu (Dataset):** `credit_card_fraud_synthetic.csv`
  * Dữ liệu tổng hợp mô phỏng các giao dịch thẻ. Biến mục tiêu là nhãn nhị phân: `0` (Normal) và `1` (Fraud).
  * **Đặc điểm cốt lõi:** Lớp dữ liệu bị mất cân bằng trầm trọng (Imbalanced Data), với số lượng giao dịch gian lận chỉ chiếm tỷ lệ rất nhỏ (khoảng 5.02%).

* **Kiểm soát rò rỉ dữ liệu (Data Leakage) & Tiền xử lý:**
  * **Chia tập dữ liệu:** Sử dụng **Stratified split** để phân chia thành 3 tập Train/Validation/Test, đảm bảo tỷ lệ giao dịch gian lận (5.02%) được duy trì ổn định và đồng đều ở mọi tập.
  * **Data Leakage Prevention:** Các bước xử lý ngoại lệ (cắt ngưỡng Q95/Q99) và chuẩn hóa dữ liệu (Scaler) được quản lý nghiêm ngặt: quá trình thiết lập tham số (`fit`) **chỉ được thực hiện trên tập Train**, sau đó mới áp dụng (`transform`) lên tập Validation và Test.

* **Chiến lược Huấn luyện & Tối ưu hóa:**
  * **Thiết lập Baseline:** Xây dựng mô hình mốc `Dummy Classifier` (dự đoán ngẫu nhiên dựa trên phân phối lớp) để làm cơ sở so sánh. Tiếp đó, đánh giá hàng loạt các mô hình cơ sở như *Logistic Regression, Decision Tree, Random Forest, Extra Trees*, và *LightGBM*.
  * **Xử lý mất cân bằng lớp:** Thay vì dùng Class Weights thông thường, dự án ứng dụng kỹ thuật **RandomOverSampler** để nhân bản các mẫu gian lận. Lưu ý: Kỹ thuật này được đóng gói khép kín và **chỉ áp dụng trên tập Train** để tránh làm sai lệch phân phối của tập đánh giá.
  * **Tối ưu siêu tham số (Hyperparameter Tuning):** Không sử dụng GridSearch thủ công, dự án tích hợp thư viện **Optuna** kết hợp với **Stratified K-Fold Cross Validation**. Phương pháp này giúp tự động hóa việc tìm kiếm không gian tham số tối ưu (đặc biệt cho mô hình LightGBM) một cách khách quan, tránh việc mô hình vô tình khớp với nhiễu của một tập validation cụ thể.

* **Tiêu chí Đánh giá & Tinh chỉnh Ngưỡng (Threshold Tuning):**
  * Loại bỏ hoàn toàn **Accuracy** (Độ chính xác tổng thể) khỏi tiêu chí đánh giá chính do tính chất lừa đảo của metric này trên dữ liệu mất cân bằng.
  * **Tối ưu F2-Score & PR-AUC:** Mô hình tập trung vào việc tối đa hóa diện tích dưới đường cong Precision-Recall (PR-AUC) và đặc biệt là **F2-Score**. F2-Score được lựa chọn vì nó đặt trọng số ưu tiên cho **Recall** cao gấp đôi Precision, phù hợp với bài toán thực tế: *Thà cảnh báo nhầm một vài giao dịch bình thường (False Positives), còn hơn bỏ lọt một giao dịch gian lận (False Negatives).*
  * **Joint Model Selection:** Kết hợp song song việc chọn mô hình tốt nhất với việc dời ngưỡng phân loại (threshold) khỏi mức mặc định 0.5, nhằm đạt được điểm F2-Score cực đại. Tiến hành **Gap Analysis** (phân tích chênh lệch hiệu suất) giữa tập Train và Test để kiểm định và ngăn chặn Overfitting.

* **Hạn chế & Hướng phát triển:** Do đặc thù là dữ liệu tổng hợp (synthetic), một số nhãn gian lận có tính ngẫu nhiên so với các biến đặc trưng. Đối với dữ liệu thực tế trong tương lai, quy trình này có thể cải tiến thêm bằng cách: trích xuất các đặc trưng chuỗi thời gian (time-series features) từ lịch sử giao dịch của từng ID người dùng, và sử dụng **SMOTE** thay vì RandomOverSampler để tạo ra các điểm dữ liệu thiểu số đa dạng hơn.
