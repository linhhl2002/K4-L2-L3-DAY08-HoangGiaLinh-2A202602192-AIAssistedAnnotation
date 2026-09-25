# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Hoàng Gia Linh

Công cụ gán nhãn đã dùng: CVAT (CVAT Docker local v2.76.0)

Báo cáo tổng kết toàn bộ quy trình Active Learning cho bài toán phát hiện xe ban đêm qua 2 vòng thực nghiệm (Vòng 0: Cold Start và Vòng 1: Fine-tune với 12 ảnh chọn lọc). Mọi số liệu được trích xuất trực tiếp từ các file `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`, `outputs/round1_diff.md` và `reports/REVIEW_LOG.csv`.

## 1. Dữ liệu và cách chia tập

Tập dữ liệu chưa gán nhãn (pool gồm 268 ảnh) và tập kiểm thử (test set gồm 20 ảnh) được chia tách theo trục thời gian với vùng đệm an toàn (buffer gồm 112 ảnh, bị loại bỏ) cách nhau tối thiểu 4.4 giây, thay vì chia ngẫu nhiên (random split).

Nguyên nhân là do video được trích xuất từ một camera đặt cố định trên cầu vượt ghi hình liên tục ở tốc độ 2.5 khung hình/giây. Trong bối cảnh camera đứng yên, các khung hình kế cận nhau (cách nhau 0.4s) có bối cảnh nền giống hệt nhau và các phương tiện lưu thông sẽ tồn tại trên khung hình trong vài giây. Nếu chia tập ngẫu nhiên, cùng một chiếc xe ở vị trí gần như không đổi sẽ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử. Điều này dẫn đến hiện tượng rò rỉ dữ liệu (data leakage) nghiêm trọng: mô hình sẽ được đánh giá trên chính những chiếc xe và góc nhìn mà nó đã được thấy khi huấn luyện. Khi đó, các chỉ số đo lường trên tập kiểm thử sẽ bị **lệch cao (lạc quan quá mức / over-optimistic)**, không phản ánh chính xác khả năng khái quát hóa (generalization) của mô hình trên các đoạn thời gian độc lập.

## 2. Mô hình khởi đầu lạnh (cold start)

Bảng chỉ số Vòng 0 (Cold Start):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg` và `outputs/metrics_round0.json`:
- **Độ lệch so với nhãn tham chiếu:** Mô hình pretrained COCO (YOLOv8n) khởi đầu với AP50 = 0.771 và độ chính xác Precision rất cao (0.925 / 92.5%), nhưng độ phủ Recall còn thấp (0.489 / 48.9%), chỉ phát hiện được 197/403 box tham chiếu và bỏ sót tới 206 box.
- **Độ phủ theo kích thước xe:** Recall đối với xe nhỏ (`R small`) chỉ đạt 0.182 (18.2% trên 66 box nhỏ), thấp hơn rất nhiều so với xe cỡ vừa (`R medium` = 0.547 trên 296 box) và xe cỡ lớn (`R large` = 0.561 trên 41 box). Điều này chỉ ra mô hình cold start gặp khó khăn nghiêm trọng nhất ở các xe ở xa sát đường chân trời (kích thước nhỏ, chỉ lộ hai chấm đèn mờ), các xe bị che khuất một phần và các xe tải lớn trong điều kiện ánh sáng đèn pha ban đêm phức tạp.
- **Trường hợp cần rà lại nhãn tham chiếu:** Nhãn tham chiếu của tập test được tạo tự động bởi mô hình và chưa qua rà soát thủ công toàn diện. Do đó, các trường hợp ánh đèn pha phản chiếu mạnh trên mặt đường ướt hoặc các xe quá xa chỉ có hai điểm sáng đèn dễ bị gán sai hoặc bỏ sót ngay trong nhãn tham chiếu. Cần có chuyên viên rà soát lại hình ảnh gốc trước khi vội kết luận mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

Chiến lược chọn mẫu sử dụng công thức tính điểm kết hợp:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
Trong đó:
- **$U$ (Uncertainty):** Độ bất định của mô hình, đo lường mức độ thiếu tự tin hoặc phân vân của dự đoán (entropy của phân phối xác suất).
- **$A$ (Ambiguity):** Độ mơ hồ, tính theo tỷ lệ các bounding box có điểm tin cậy nằm trong vùng lấp lửng (khoảng tự tin trung bình, khó dứt khoát là xe hay không phải xe).
- **$D$ (Diversity):** Độ đa dạng không gian, đo lường sự phân bố rải rác của các box trên toàn bộ khung hình thay vì co cụm một chỗ.
- **`MIN_GAP_S`:** Ngưỡng khoảng cách thời gian tối thiểu giữa hai khung hình liên tiếp được chọn (ví dụ 2.0s). Vai trò của `MIN_GAP_S` là ngăn chặn việc chọn các khung hình gần trùng lặp (near-duplicates) có bối cảnh gần như y hệt nhau, từ đó tối ưu hóa công sức gán nhãn và tăng cường tính đa dạng cho tập train.

**Minh chứng từ các frame:**
- `frame_0182.jpg` (Rank 1, score = 0.9591, U = 0.9182, A = 1.0, D = 1.0): Điểm bất định và mơ hồ tuyệt đối do xuất hiện 18/28 box ở mức tin cậy trung bình dưới ánh sáng đèn phức tạp.
- `frame_0331.jpg` (Rank 5, score = 0.9154, 47 box, U = 0.8308, A = 1.0): Mật độ xe dày đặc, đòi hỏi chi phí gán nhãn cao nhưng mang lại nhiều thông tin giá trị về việc tách cụm xe sát nhau.
- `frame_0099.jpg` (Rank 8, score = 0.9063, U = 0.9460, A = 0.7778): Điểm $U$ cao vượt trội (0.9460) do xuất hiện nhiều xe con ở làn đối diện phía xa.
- `frame_0372.jpg` (Rank 6, score = 0.9101, `selected == False`): Mặc dù có điểm số rất cao (0.9101), frame này đã bị thuật toán lọc bỏ vì chỉ cách `frame_0369.jpg` (Rank 2, mốc 147.6s) đúng 1.2s (< `MIN_GAP_S`). Đây là minh chứng rõ ràng cho việc cân nhắc đa dạng hóa bối cảnh và tránh lãng phí chi phí gán nhãn.

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?**
Không. Điểm bất định cao chỉ phản ánh rằng mô hình hiện tại đang gặp khó khăn và thiếu tự tin trước bức ảnh đó. Nó hoàn toàn không đảm bảo rằng việc gán nhãn bức ảnh đó sẽ giúp cải thiện mô hình sau khi huấn luyện. Nếu lượng mẫu được bổ sung quá ít hoặc phương pháp fine-tune chưa tối ưu, mô hình có thể bị quá khớp hoặc phá vỡ các đặc trưng tổng quát đã học.

## 4. Các vòng học chủ động (active learning)

Bảng so sánh tổng hợp các vòng thực nghiệm từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 287 | 0.358 | -0.413 | 1.000 | 0.005 | 0.010 | 0.000 | 0.007 | 0.000 |

### Chi tiết mức độ chỉnh sửa nhãn Vòng 1 (từ `outputs/round1_diff.md`):
- Tổng cộng 12 ảnh: Mô hình ban đầu đề xuất 169 box, sau khi rà soát và chỉnh sửa trên CVAT đã thu được 287 box chuẩn xác.
- **Accepted (Giữ nguyên):** 150 box (tỷ lệ chấp nhận đạt 89% số box đề xuất).
- **Edited (Chỉnh sửa):** 9 box (điều chỉnh co box ôm sát thân xe, cắt bỏ vệt đèn pha như ở `frame_0107.jpg`, `frame_0182.jpg`).
- **Deleted (Xóa bỏ - False Positive của AI):** 10 box (loại bỏ các box AI nhận nhầm vệt phản quang ánh đèn trên mặt đường ướt như ở `frame_0312.jpg`).
- **Added (Bổ sung - False Negative của AI):** 128 box (bổ sung các xe ở xa bị bỏ sót và tách các box gộp xe sát nhau như ở `frame_0099.jpg`, `frame_0331.jpg`).

### Biến động chỉ số và phân tích hiện tượng sụt giảm:
- **Chỉ số AP50:** Sụt giảm mạnh từ 0.771 xuống 0.358 ($\Delta \text{AP50} = -0.413$, giảm 53.5% tương đối).
- **Precision & Recall:** Precision tăng lên mức tuyệt đối 1.000 (100%), tuy nhiên Recall bị sụp đổ từ 0.489 xuống còn 0.005 (0.5%), F1 giảm từ 0.640 xuống 0.010. Mô hình chỉ phát hiện được đúng 2/403 box trên tập test (TP=2, FP=0, FN=401).
- **Phân rã theo kích thước:** `R small` giảm từ 0.182 về 0.000; `R large` giảm từ 0.561 về 0.000; `R medium` giảm từ 0.547 về 0.007.

**Phân tích nguyên nhân kỹ thuật (Catastrophic Forgetting & Overfitting):**
Mô hình YOLOv8n được fine-tune trong 50 epochs trên một tập dữ liệu cực kỳ nhỏ (chỉ 12 ảnh với 287 box) mà không đóng băng (freeze) các tầng backbone và giữ nguyên learning rate mặc định. Quá trình này đã dẫn tới hiện tượng **quên thảm họa (catastrophic forgetting)**: các trọng số phát hiện đa dạng vật thể từ tập COCO đồ sộ đã bị ghi đè bởi phân phối hẹp của 12 ảnh. Mô hình bị quá khớp (overfitting) và trở nên **quá thận trọng (overly conservative)** — nó chỉ đưa ra dự đoán khi độ tự tin gần như 100%, dẫn đến việc hoàn toàn không có dự đoán sai (Precision = 1.000, FP = 0) nhưng lại bỏ qua hầu như toàn bộ các xe khác trên tập kiểm thử (Recall = 0.005, FN = 401). Đây là một hiện tượng kinh điển và là bài học thực tế vô cùng quan trọng khi triển khai Active Learning trong các vòng đầu tiên.

### Đối chiếu quan sát, lỗi pre-label và ca khó:
- Dựa vào `reports/REVIEW_LOG.csv`, `round1_diff.md` và `compare_round1.jpg`: Mô hình sau khi train đã khắc phục triệt để lỗi sinh ra box giả ở các vệt sáng mặt đường (đã xóa 10 box rác ở bước gán nhãn), nhưng lại làm mất khả năng phát hiện các xe ở xa.
- **Ca khó theo guideline:** Tình huống hai xe con đi sát nhau ở làn đối diện ban đêm (`frame_0331.jpg`). AI pre-label ban đầu chỉ vẽ một box lớn bao trùm cả hai xe. Tuân thủ `GUIDELINE_LABEL.md`, người gán nhãn đã tách thành hai box riêng biệt ôm sát từng thân xe dựa trên sự phân tách của các cụm đèn pha.

## 5. Kết luận và giới hạn

### Đánh giá kết quả và quyết định:
Kết quả Vòng 1 ghi nhận sự sụt giảm AP50 và Recall do hiện tượng quá khớp trên tập dữ liệu nhỏ 12 ảnh. Tuy nhiên, quá trình này giúp làm sạch nhãn và làm sáng tỏ cơ chế học của mô hình. Quyết định là **tiếp tục thực hiện Vòng 2** nhưng cần điều chỉnh chiến lược huấn luyện thay vì dừng lại.

### Đề xuất 2 ca còn yếu/bất định cho vòng tiếp theo:
1. **Ca xe con ở khoảng cách xa (Small vehicles):** Các frame ở phân đoạn $t \in [30\text{s}, 60\text{s}]$ (tương tự `frame_0099.jpg`, `frame_0107.jpg`) nhằm cứu vãn chỉ số `R small` (đang ở mức 0.000). Chi phí gán nhãn trung bình (khoảng 20-30 box/ảnh), nguy cơ trùng lặp thấp nếu duy trì khoảng cách tối thiểu $\ge 3$ giây.
2. **Ca xe tải lớn / xe container ban đêm (Large vehicles):** Các frame ở phân đoạn $t \in [70\text{s}, 90\text{s}]$ (tương tự `frame_0182.jpg`, `frame_0227.jpg`) để phục hồi chỉ số `R large` (đang ở mức 0.000). Chi phí gán nhãn thấp đến trung bình (khoảng 15-25 box/ảnh).

### Giới hạn của thực nghiệm:
- Tập kiểm thử chỉ có 20 ảnh (403 box tham chiếu), độ biến thiên thống kê lớn.
- Quy tắc loại bỏ các box có chiều cao $< 16$ pixel giúp tránh phạt mô hình ở các trường hợp quá mơ hồ, nhưng cũng giới hạn việc đánh giá năng lực phát hiện ở tầm cực xa.
- Nhãn tham chiếu test do mô hình tạo tự động, chưa được con người thẩm định 100%, có thể chứa thiên kiến cố hữu.

### Các hạng mục cần kiểm tra và cải tiến trước khi train thêm:
1. **Đóng băng Backbone (Freeze Layers):** Đóng băng 10 tầng đầu của backbone YOLOv8 khi fine-tune để bảo toàn các đặc trưng thị giác cơ bản đã học từ COCO, chỉ cập nhật các tầng detection head.
2. **Điều chỉnh siêu tham số:** Giảm learning rate ban đầu (`lr0`) và giảm số epochs huấn luyện (từ 50 xuống 15-20 epochs) để tránh overfitting trên số lượng mẫu nhỏ.
3. **Chiến lược Replay / Tích lũy dữ liệu:** Giữ lại và kết hợp toàn bộ dữ liệu của Vòng 1 khi nạp thêm dữ liệu Vòng 2 (cumulative training) để mở rộng dung lượng mẫu và ổn định gradient.
