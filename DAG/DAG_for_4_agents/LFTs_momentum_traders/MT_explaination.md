# BẢN GIẢI TRÌNH CƠ CHẾ NHÂN QUẢ  
## Tác nhân giao dịch theo xu hướng (Momentum Traders - LFT)

---

## 1. Tổng quan về vai trò của Momentum Traders

Trong hệ thống vi cấu trúc thị trường, **Momentum Traders** thuộc nhóm **Giao dịch Tần suất Thấp (LFT)**, đóng vai trò là người tiêu thụ thanh khoản chủ động (*Aggressive Takers*).

Điểm khác biệt cốt lõi so với Noise Traders:
- Momentum Traders là các thuật toán **có trí nhớ**

Cơ chế kỳ vọng:
\[
\text{Trend Extrapolation}
\]

→ Giá trong quá khứ sẽ tiếp tục trong tương lai  

---

Khi thị trường rung lắc:
- Thuật toán bám xu hướng + Stress  
→ biến Momentum thành:

> **“Can xăng” khuếch đại lực bán**

→ Góp phần gây ra **Flash Crash**

---

## 2. Bảng chú giải tín hiệu (Color Legend & Visual Semantics)

### 🔴 Biến can thiệp (Treatment - X)
- Stress sinh lý ($\sigma$)  
→ Nguồn gốc sai lệch hành vi (ATE)

---

### 🟢 Biến kết quả (Outcome - Y)
- Khối lượng thực khớp  
- Giá đóng cửa  

---

### ⚪ Biến nhiễu (Confounders - Z)
- $W_{t-1}, I_{t-1}, Vol_{t-1}$  
→ Cần control để tránh bias

---

### 🟡 Biến ngoại sinh / dị chất
- $\xi_i$, $\eta_L$  
→ Tạo **Heterogeneity**  
→ Gây **Cascading Failure**

---

### 🔵 Biến trung gian (Mediators)
- $\alpha$, $\gamma$, $\lambda$

---

### 🟣 Market State & Memory
- $P_t$, $LOB_t$, $\hat{P}_t$

\[
\hat{P}_t = \text{Moving Average}
\]

→ Biến ký ức nội sinh

---

### 🩵 Tính toán nội tại
- $E_{trend,t}$, $\Omega$

---

### ⚫ Gates & Blackbox
- $I_{thr}$ + Signal Ignore  
→ Ngắt causal flow

---

## 3. Giai đoạn t-1: Nguồn gốc hoảng loạn

### Dị chất ($\xi_i$)
- Mỗi trader có tolerance khác nhau  
→ Tạo phân tán hành vi

---

### Mark-to-Market ($\pi_{t-1}$)
- Từ:
  - $I_{t-1}$
  - $P_t$

---

### Stress ($\sigma_{t-1}$)
\[
\sigma_{t-1} = f\left(\frac{\pi_{t-1}}{W_{t-1}}, \xi_i\right)
\]

---

### Solvency Gate
\[
W_{t-1} \le 0
\]

→ Bị loại khỏi hệ thống

---

## 4. Giai đoạn t: Xu hướng + sụp đổ nhận thức

### Hình thành tín hiệu xu hướng

\[
P_t < \hat{P}_t \Rightarrow E_{trend,t} < 0
\]

→ Kích hoạt **bán khống**

---

### Cổng mù nhận thức ($I_{thr}$)

- Stress ↑ → Cognitive Overload  
- Ngưỡng:
\[
I_{thr}(\sigma)
\]

→ Bỏ qua tín hiệu nhỏ  
→ Chỉ phản ứng khi **giá sập mạnh**

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

### Hawkes process
\[
\lambda_t \uparrow
\]

→ Spam lệnh bán  
→ Quote stuffing

---

## 5. Cú kết liễu: Adverse Selection

### Luồng bán tháo
\[
Demand_t
\]

---

### Độ trễ vật lý
\[
\eta_L \gg \eta_H
\]

---

### Adverse Selection
- HFT nhìn thấy trước  
→ rút toàn bộ Bid

---

### Outcome

- Khối lượng thực khớp ≈ 0  
- Giá sập mạnh:

\[
P_{t+1} \downarrow\downarrow
\]

---

### Feedback Loop

\[
P_{t+1} \downarrow 
\Rightarrow \hat{P}_{t+1} \downarrow 
\Rightarrow \pi \uparrow 
\Rightarrow \sigma \uparrow 
\Rightarrow \text{Bán tháo tiếp}
\]

---

## 6. Kết luận

Momentum Traders không chỉ theo xu hướng, mà:

> **Khuếch đại xu hướng thành sụp đổ**

Flash Crash là kết quả của:
- Trend-following algorithm  
- + Stress tâm lý  
- + Feedback loop  

→ Tạo **cascading crash**