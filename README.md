# Bài toán question answer trong cuộc thi Zalo AI Challenge
## Thực tập kĩ thuật : @ngovietbach
### Cài đặt lại từ repo gốc : https://github.com/trangtv57/QAZalo
#### Cấu trúc thư mục 
1. bert_model : chứa 3 file bao gồm __init__.py , bert.py , bert_utils.py
  - bert_utils.py: tải mô hình bert đã được huấn luyện trước : PhoBert và tạo một bộ phân loại
  - bert.py : mô hình tùy chỉnh "BERT_QA" được thiết kế cho nhiệm vụ question answer
2. module_dataset : Định nghia một số lớp và hàm để xử lý dữ liệu và chuẩn bị dữ liệu đầu vào cho mô hình BERT
  - Lớp SquadExample : biểu diễn ví dụ trong tập dữ liệu SQuAD
  - Lớp InputFeatures : biểu diễn đặc trưng đầu vào của mô hình bert 
  - read examples from file : Đọc ví dụ từ một tệp đầu vào và tạo ra các đối tượng SquadExample
  - convert examples to features : chuyển đổi các ví dụ thành đặc trưng đầu vào cho mô hình bert 
  - make_dataset : sắp xếp dữ liệu và chia thành các batch 
  - eda_dataset: hàm này thực hiện phân tích dữ liệu thô để hiểu thêm về phân phối của các lớp (câu trả lời có hoặc không có) trong tập dữ liệu SQuAD. Nó in ra tổng số câu trả lời có và không có.
3. schedule.py : Lịch trình tốc độ học tập này giúp cân nhắc việc tăng dần tốc độ học tập ban đầu để tránh việc mô hình bị "đói học" (underfitting) trong giai đoạn đầu, và sau đó giảm dần tốc độ học tập để ổn định quá trình huấn luyện và tránh bị "quá khớp" (overfitting).
4. metrics.py : Xây dựng độ đo trong quá trình đào tạo và đánh giá mô hình học máy 
  - Các lớp metric này được xây dựng để tính toán các giá trị thống kê liên quan đến hiệu suất của mô hình học máy trong việc dự đoán nhãn (true/false positives/negatives) và các độ đo như độ nhớ, độ chính xác, và F1 score.
  - Các metric này có thể được sử dụng trong quá trình đào tạo và đánh giá mô hình để theo dõi và đánh giá hiệu suất của mô hình trên dữ liệu thực tế.
5. utils.py : một tập hợp các hàm và lớp được sử dụng để đào tạo và đánh giá mô hình
  - train: Hàm này dùng để đào tạo mô hình QA dựa trên dữ liệu đào tạo. Nó sử dụng tối ưu hóa được tạo ra bởi create_optimizer và thực hiện việc đào tạo mô hình qua nhiều epochs. Hàm này cũng tính toán các độ đo đào tạo như độ mất mát, F1 score và độ chính xác.
  - train_step: Hàm này thực hiện một bước đào tạo trong quá trình đào tạo mô hình. Nó tính toán mất mát và cập nhật trọng số mô hình thông qua lan truyền ngược (backpropagation).
  - evaluate: Hàm này sử dụng để đánh giá mô hình trên dữ liệu đánh giá (validation dataset). Nó tính toán mất mát, F1 score và độ chính xác trên dữ liệu đánh giá.
6. train.py : Trong ngữ cảnh của đào tạo mô hình máy học và học sâu, "epochs" (có thể dịch là "vòng lặp đào tạo" hoặc "chu kỳ đào tạo") đề cập đến số lần mô hình được đào tạo trên toàn bộ tập dữ liệu huấn luyện. Mỗi epoch tương đương với việc mô hình được đưa qua toàn bộ dữ liệu huấn luyện một lần.
Khi đào tạo một mô hình máy học, quá trình đào tạo thường diễn ra qua nhiều epochs để mô hình có cơ hội học và cải thiện từng lần lặp. Mỗi epoch bao gồm các bước sau:
  - Sáo trộn dữ liệu: Dữ liệu huấn luyện thường được sắp xếp theo một thứ tự cố định. Trong mỗi epoch, dữ liệu thường được sắp xếp lại (sáo trộn) để đảm bảo rằng mô hình không học theo một thứ tự cụ thể và có khả năng tổng quát hóa tốt hơn.
  - Chia thành batch: Dữ liệu sau khi được sắp xếp lại được chia thành các "lô" (batch). Một lô là một tập hợp con của dữ liệu huấn luyện, và mô hình sẽ được đào tạo trên từng lô một.
  - Đào tạo trên từng batch: Mô hình sẽ được đào tạo trên từng lô dữ liệu một. Trong quá trình này, mô hình sẽ tính toán mất mát và cập nhật trọng số dựa trên mất mát đó.
  - Tính toán kết quả đào tạo: Sau khi hoàn thành một epoch (đào tạo qua toàn bộ dữ liệu), có thể tính toán kết quả của mô hình trên tập dữ liệu đánh giá (nếu có). Điều này giúp theo dõi hiệu suất của mô hình sau mỗi vòng lặp đào tạo.
  - Số lượng epochs là một tham số quan trọng trong quá trình đào tạo mô hình và thường được xác định trước. Việc chọn số lượng epochs phù hợp có thể ảnh hưởng đến khả năng học của mô hình và tránh tình trạng "quá đào tạo" (overfitting) hoặc "chưa đủ đào tạo" (underfitting).
##### Running 
* model in save/weights
* Run file train.py
* f1 score : 0.42
* ![Image](/assets/img/image1.png "image")



