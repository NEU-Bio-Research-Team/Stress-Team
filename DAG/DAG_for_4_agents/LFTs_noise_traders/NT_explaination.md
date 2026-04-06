# BẢN GIẢI TRÌNH CƠ CHẾ NHÂN QUẢ  
## Tác nhân giao dịch nhiễu (Noise Traders - LFT)

---

## 1. Tổng quan về vai trò của Noise Traders

Trong hệ sinh thái mô phỏng vi cấu trúc thị trường, **Noise Traders** thuộc nhóm **Giao dịch Tần suất Thấp (Low-Frequency Traders - LFT)**, đóng vai trò là người tiêu thụ thanh khoản chủ động (*Aggressive Takers*).

Khác với **Momentum Traders**, Noise Traders hình thành kỳ vọng dựa trên các cú sốc ngẫu nhiên:
\[
N(0,1)
\]

Vai trò vĩ mô:
- Tạo ra "tia lửa" biến động (*stochasticity*) ban đầu cho thị trường  

Dưới áp lực tâm lý, kết hợp với bất lợi về độ trễ vật lý:
\[
\eta_L
\]

→ Noise Traders trở thành nạn nhân trực tiếp của:
- **Adverse Selection**  
- Góp phần gây ra **Flash Crash**

---

## 2. Bảng chú giải tín hiệu (Color Legend & Visual Semantics)

### 🔴 Biến can thiệp (Treatment - X)
- Đại diện: Stress sinh lý ($\sigma$)  
- Ý nghĩa: Trung tâm causal → đo **ATE**

---

### 🟢 Biến kết quả (Outcome - Y)
- Đại diện: Khối lượng thực khớp, giá đóng cửa  
- Ý nghĩa: Systemic event

---

### ⚪ Biến nhiễu (Confounders - Z)
- Đại diện: $W_{t-1}, I_{t-1}, Vol_{t-1}$  
- Ý nghĩa:
  - Nếu không control → **Omitted Variable Bias**

---

### 🟡 Biến ngoại sinh / dị chất
- Đại diện: $\xi_i$, nhiễu, $\eta_L$  
- Ý nghĩa:
  - Tạo **Heterogeneity**
  - Gây **Cascading Failure**

---

### 🔵 Biến trung gian (Mediators - M)
- $\alpha$, $\gamma$, $\lambda$  
→ Mechanism truyền dẫn

---

### 🟣 Trạng thái thị trường
- LOB, giá hiện tại

---

### 🩵 Tính toán nội tại
- $E_{noise,t}$, $\Omega$, Demand

---

### ⚫ Gates & Blackbox
- Solvency + Signal Ignore  
→ Ngắt causal chain

---

## 3. Giai đoạn t-1: Cội nguồn hoảng loạn

### Dị chất tác nhân ($\xi_i$)
- Mỗi trader có mức chịu rủi ro khác nhau  
→ Tạo phân tán hành vi (Heterogeneity)

---

### Mark-to-Market ($\pi_{t-1}$)
- Phụ thuộc:
  - $I_{t-1}$
  - Giá thị trường  

→ Tạo tổn thất trạng thái

---

### Stress sinh lý ($\sigma_{t-1}$)
\[
\sigma_{t-1} = f\left(\frac{\pi_{t-1}}{W_{t-1}}, \xi_i\right)
\]

→ Nguồn gốc hoảng loạn

---

### Solvency Gate
\[
W_{t-1} \le 0
\]

→ Loại khỏi hệ thống

---

## 4. Giai đoạn t: Đóng băng nhận thức

Noise Trader hình thành:
\[
E_{noise,t}
\]

Nhưng bị bóp méo bởi Stress qua các cơ chế:

---

### Cổng mù nhận thức ($I_{thr}$)
- Stress cao → **Cognitive Overload**
- Ngưỡng chú ý giãn rộng:
\[
I_{thr}(\sigma)
\]

→ Bỏ qua tín hiệu yếu  
→ **Cognitive Freeze**

---

### Giảm độ xông xáo
\[
\alpha_t \downarrow
\]

---

### Tăng hủy lệnh
\[
\gamma_t \uparrow
\]

---

### Hawkes process (quan trọng)
\[
\lambda_t \uparrow
\]

→ Self-exciting  
→ Spam lệnh → Nhiễu hệ thống

---

## 5. Cú kết liễu: Adverse Selection

### Luồng cầu
\[
Demand_t
\]

→ đi vào:
- LOB  
- Market State

---

### Độ trễ vật lý
\[
\eta_L \gg \eta_H
\]

→ Lệnh đến chậm

---

### Adverse Selection
- Market Makers (HFT) thấy trước  
→ rút thanh khoản

---

### Outcome

- Khối lượng thực khớp ≈ 0  
- Giá trượt sâu:

\[
P_{t+1} \downarrow\downarrow
\]

→ Khuếch đại **Flash Crash**

---

## 6. Kết luận

Noise Traders không phải nguyên nhân chính gây crash, mà là:

> **Tác nhân khuếch đại (amplifier)**

Cơ chế:
- Stress → hành vi méo  
- Hành vi méo → spam lệnh  
- Spam + delay → không khớp  
- Không khớp → giá rơi  

→ Tạo vòng lặp sụp đổ