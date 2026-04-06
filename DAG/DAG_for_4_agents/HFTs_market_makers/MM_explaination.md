# BẢN GIẢI TRÌNH CƠ CHẾ NHÂN QUẢ  
## Thuật toán tạo lập thị trường tần suất cao (HFT Market Makers)

---

## 1. Tổng quan về vai trò của Market Makers (HFT)

Trong hệ thống tài chính vi mô, **Market Makers (MM)** hay **HFT** đóng vai trò là người cung cấp thanh khoản thụ động (*Passive Liquidity Providers*).  

Khác với nhóm **LFT (Noise/Momentum Traders)** là những người chủ động lấy thanh khoản (*Takers*), MM sinh lời bằng cách ăn chênh lệch giá mua - bán (*Bid-Ask Spread*) thay vì dự đoán xu hướng giá dài hạn.

Vì bản chất thụ động và sở hữu siêu tốc độ ($\eta_H \ll \eta_L$), cấu trúc nhân quả của MM khi đối mặt với rủi ro tâm lý có thể gây ra hiện tượng:

- **Bốc hơi thanh khoản (Liquidity Vacuum)**  
→ Nguyên nhân trực tiếp dẫn đến **Flash Crash**

Điều này cho thấy thị trường tài chính không chỉ là hệ thống thuật toán, mà là một:

> **Hệ sinh thái sinh học - kỹ thuật liên hợp (bio-technical ecosystem)**

---

## 2. Bảng chú giải tín hiệu hình ảnh (Color Legend & Visual Semantics)

Sơ đồ DAG được mã hóa màu theo **Causal Inference**:

### 🔴 Biến can thiệp (Treatment - X)
- Đại diện: Stress sinh lý ($\sigma$)  
- Ý nghĩa: Biến trung tâm → đo **ATE (Average Treatment Effect)**

### 🟢 Biến kết quả (Outcome - Y)
- Đại diện: Thanh khoản thực tế, giá đóng cửa  
- Ý nghĩa: Điểm cuối → giải thích **Flash Crash**

### ⚪ Biến nhiễu (Confounders - Z)
- Đại diện: $W_{t-1}, I_{t-1}, Vol_{t-1}$  
- Ý nghĩa:  
  - Nếu không kiểm soát → **Omitted Variable Bias**

### 🟡 Biến ngoại sinh / dị chất
- Đại diện: $\xi_i$, nhiễu ngẫu nhiên, $\eta_H$  
- Ý nghĩa:  
  - Tạo **Heterogeneity**  
  - Gây **Cascading Failure**

### 🔵 Biến trung gian (Mediators - M)
- $\Delta x$, $\gamma$, $\alpha$, $\lambda$  
→ Cơ chế truyền tác động

### 🟣 Trạng thái thị trường
- LOB, sự kiện vi mô, giá hiện tại

### 🩵 Tính toán nội tại
- Kỳ vọng, ràng buộc tồn kho ($\Omega$)

### ⚫ Solvency Gate
- Điều kiện: $W \le 0$  
→ Ngắt toàn bộ causal chain

---

## 3. Giai đoạn t-1: Nguồn gốc hoảng loạn & dị chất

### Dị chất tác nhân ($\xi_i$)
- Đại diện: mức chịu rủi ro khác nhau  
- Giải thích:
  - MM yếu → sụp trước  
  - MM mạnh → sụp sau  
→ **Cascading Failure**

### Mark-to-Market ($\pi_{t-1}$)
- Phụ thuộc:
  - Tồn kho $I_{t-1}$
  - Biến động $Vol_{t-1}$
  - Giá $P_t$

### Stress sinh lý ($\sigma_{t-1}$)
- Theo **Prospect Theory**:
\[
\sigma_{t-1} = f\left(\frac{\pi_{t-1}}{W_{t-1}}, \xi_i\right)
\]

→ **Biến Treatment trung tâm**

### Solvency Gate
- Nếu:
\[
W_{t-1} \le 0
\]
→ Agent bị loại khỏi hệ thống

---

## 4. Giai đoạn t: Cơ chế phòng vệ độc hại

Stress kích hoạt 4 mediator:

### 1. Mở rộng biên độ
\[
\Delta x_t \uparrow
\]
→ Lệnh bị đẩy xa → LOB rỗng

### 2. Giảm độ xông xáo
\[
\alpha_t \downarrow
\]

### 3. Tăng hủy lệnh
\[
\gamma_t \uparrow
\]
→ Thanh khoản ảo (Spoofing)

### 4. Quá trình Hawkes (quan trọng nhất)
\[
\lambda_t \uparrow
\]

→ Self-exciting process  
→ Quote stuffing  
→ Nghẽn hệ thống

---

## 5. Đặc quyền thuật toán

### Kích hoạt theo sự kiện
\[
A_{H,t}
\]

### Ràng buộc kép
- ≤ 25% khối lượng LFT  
- Bị giới hạn bởi tồn kho ($\Omega$)

### Kỳ vọng bằng 0
\[
E = 0
\]

→ Lấy $P_t$ làm trung tâm

---

## 6. Cú kết liễu: Flash Crash (t+1)

### Lợi thế tốc độ
\[
\eta_H \ll \eta_L
\]

### Adverse Selection
- MM thấy trước lệnh bán lớn  
→ rút thanh khoản

### Liquidity Vacuum
- MM rút lệnh trong vài **milliseconds**

### Outcome
- Lệnh bán đâm vào khoảng trống  
→ Giá sập mạnh:
\[
P_{t+1} \downarrow\downarrow
\]

---

## 7. Kết luận

Flash Crash **không đến từ panic của trader**, mà đến từ:

> **Cơ chế tự phòng vệ của Market Makers**

Feedback loop:
- Biến động ↑ → Stress ↑  
- Stress ↑ → Rút thanh khoản  
- Rút thanh khoản → Biến động ↑  

→ Tạo **tipping point** → sập thị trường