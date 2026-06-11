# Bài Tập Lớn Cuối Kỳ Môn Học Máy (Machine Learning Final Project)

Repository này lưu trữ mã nguồn, dữ liệu và báo cáo kết quả thực hiện bài tập lớn cuối kỳ môn Học máy. Dự án bao gồm hai bài toán kinh điển trong Machine Learning: **Hồi quy tuyến tính (Linear Regression)** và **Phân loại nhị phân (Binary Classification)**.

## 📂 Cấu trúc Repository

- `BTL_HoiQuyTuyenTinh_final.ipynb`: Jupyter Notebook thực hiện phân tích và huấn luyện mô hình cho bài toán dự đoán giá nhà.
- `BTL_PhanLoaiNhiPhan_final.ipynb`: Jupyter Notebook thực hiện xử lý dữ liệu và huấn luyện mô hình cho bài toán phát hiện giao dịch gian lận.
- `USA_Housing.csv`: Bộ dữ liệu sử dụng cho bài toán hồi quy.
- `credit_card_fraud_synthetic.csv`: Bộ dữ liệu tổng hợp sử dụng cho bài toán phân loại.

---

## 🚀 Chi tiết các bài toán

### 1. Dự đoán giá nhà (Hồi quy tuyến tính)
- **Mục tiêu:** Xây dựng mô hình học máy để dự đoán giá nhà (`Price`) dựa trên các đặc điểm khu vực.
- **Dataset:** `USA_Housing.csv`
  - Các biến đầu vào (Features): Thu nhập trung bình khu vực (`Avg. Area Income`), Tuổi trung bình của nhà (`Avg. Area House Age`), Số phòng trung bình (`Avg. Area Number of Rooms/Bedrooms`), Dân số khu vực (`Area Population`).
  - Biến mục tiêu (Target): `Price` (biến liên tục).
- **Phương pháp & Mô hình:**
  - Áp dụng các mô hình: Linear Regression, Ridge Regression và ElasticNet.
  - Sử dụng Cross-Validation (`CV_RMSE_Mean`) để đánh giá thay vì chỉ phụ thuộc vào tập test.
- **Kết quả:** **Ridge Regression** là mô hình cho hiệu suất ổn định và tốt nhất trong quá trình Cross-validation, giúp giảm thiểu hiện tượng quá khớp (overfitting) so với hồi quy tuyến tính thông thường.

### 2. Phát hiện gian lận thẻ tín dụng (Phân loại nhị phân)
- **Mục tiêu:** Phân loại các giao dịch thẻ tín dụng thành 2 nhóm: Bình thường (Normal) hoặc Gian lận (Fraudulent).
- **Dataset:** `credit_card_fraud_synthetic.csv` (Dữ liệu giao dịch tổng hợp).
- **Phương pháp & Kỹ thuật nổi bật:**
  - **Data Splitting:** Sử dụng Stratified split để duy trì tỷ lệ lớp thiểu số (gian lận) ổn định giữa các tập Train/Val/Test.
  - **Feature Engineering & Preprocessing:** Xử lý các đặc trưng có kiểm soát rò rỉ dữ liệu (data leakage) nghiêm ngặt (chỉ fit các ngưỡng như Q95/Q99 hoặc Scaler trên tập Train).
  - **Tối ưu siêu tham số (Hyperparameter Tuning):** Ứng dụng `Optuna` kết hợp với Stratified K-Fold CV để tìm kiếm bộ tham số tối ưu, tránh việc chọn mô hình dựa trên nhiễu (noise) của single validation set.
  - **Đánh giá mô hình:** Thực hiện Joint model selection và Threshold tuning tập trung vào **F2-Score** (nhằm ưu tiên Recall - bắt được càng nhiều giao dịch gian lận càng tốt) thay vì chỉ đánh giá Accuracy.
