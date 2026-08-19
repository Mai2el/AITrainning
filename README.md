# 🛡️ DDoS Detection using Deep Learning

> **Ứng dụng Deep Learning trong việc nhận dạng và phát hiện tấn công từ chối dịch vụ phân tán (DDoS)**

Đồ án chuyên ngành – **Trường Đại học Công nghệ Thông tin, ĐHQG TP.HCM**

**Sinh viên thực hiện:**

* Nguyễn Huỳnh Nhân – 23521080
* Võ Hoài Nam – 23520990

**Năm:** 2026

---

## 📌 Giới thiệu

DDoS (Distributed Denial of Service) là một trong những mối đe dọa phổ biến đối với các hệ thống mạng hiện nay. Việc phát hiện DDoS gặp nhiều khó khăn do lưu lượng mạng có số lượng đặc trưng lớn, phân phối dữ liệu không đồng nhất và sự khác biệt giữa lưu lượng hợp lệ với lưu lượng tấn công ngày càng khó nhận biết.

Đồ án nghiên cứu và xây dựng một mô hình **Deep Learning dành cho dữ liệu mạng dạng bảng (Tabular Network Data)** nhằm nhận dạng và phân loại các hành vi DDoS.

Giải pháp tập trung vào hai thành phần chính:

* **Feature-wise Tokenization** – chuyển đổi từng đặc trưng của network flow thành token.
* **Parallel Multi-Branch Attention** – sử dụng nhiều nhánh Attention hoạt động song song để học các mối quan hệ phức tạp giữa các đặc trưng mạng.

Mục tiêu là xây dựng một hệ thống có khả năng:

1. Phân biệt lưu lượng **Benign / Attack**.
2. Phân loại chi tiết các loại tấn công DDoS.
3. Khai thác các tương tác phi tuyến giữa các đặc trưng network flow.
4. Giảm ảnh hưởng của sự mất cân bằng dữ liệu trong quá trình huấn luyện.

---

## 🧠 Kiến trúc hệ thống

Hệ thống được thiết kế thành hai phân hệ chính:

```text
                    Network Flow Dataset
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing   │
                 │                     │
                 │ • Cleaning         │
                 │ • Feature Selection│
                 │ • Quantile Profiling│
                 │ • Grouping          │
                 │ • Tokenization      │
                 └──────────┬──────────┘
                            │
                     Token Matrix
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Tabular Embedding   │
                 │                     │
                 │ PLE Value Embedding│
                 │ Feature Embedding  │
                 └──────────┬──────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │ Parallel Multi-Branch        │
             │ Multi-Head Attention         │
             │                              │
             │ Branch 1 ──► MHA             │
             │ Branch 2 ──► MHA             │
             │ Branch 3 ──► MHA             │
             │ Branch 4 ──► MHA             │
             └──────────────┬───────────────┘
                            │
                            ▼
                    Multi-view Pooling
                            │
                            ▼
                         MLP
                            │
                            ▼
                  DDoS Classification
```

Kiến trúc sử dụng **4 nhánh Attention × 4 attention heads = 16 heads**, được xếp chồng qua 4 tầng Transformer. Vector embedding có kích thước 256, được tạo từ PLE value embedding 192 chiều và feature identity embedding 64 chiều.

---

## 🔄 Data Preprocessing Pipeline

Quy trình tiền xử lý gồm nhiều bước nhằm giảm nhiễu và đưa dữ liệu network flow về dạng phù hợp với mô hình Deep Learning.

### 1. Data Cleaning

Các giá trị `+inf` và `-inf` phát sinh từ quá trình trích xuất network flow được xử lý nhằm tránh ảnh hưởng đến quá trình huấn luyện.

Các trường có nguy cơ gây **data leakage**, chẳng hạn:

* Source IP
* Destination IP
* Timestamp
* Source Port
* Raw Label

được loại bỏ.

Các biến thể `Benign-*` được gộp thành lớp `Benign`, trong khi nhóm `Suspicious` được loại bỏ.

### 2. Hybrid Feature Selection

Dataset ban đầu có hơn **300 network-flow features**.

Hệ thống sử dụng ba phương pháp để lựa chọn đặc trưng:

* **ANOVA F-value**
* **Information Gain / Mutual Information**
* **Extra Trees Feature Importance**

Các kết quả được kết hợp thông qua cơ chế **Consensus + Hybrid Score** để lựa chọn tập đặc trưng quan trọng cho mô hình.

### 3. Quantile Profiling

Mỗi đặc trưng được phân tích dựa trên các phân vị và thống kê phân phối như:

* Quantiles
* Tail ratio
* Spread ratio
* Zero ratio
* Number of unique values
* Skewness
* Kurtosis

Các thông tin này được sử dụng để xác định chiến lược biến đổi phù hợp cho từng feature.

### 4. Feature Grouping

Các feature được phân thành các nhóm:

```text
Port
Binary
Low-cardinality
Skewed
Normal
```

Việc phân nhóm giúp áp dụng phương pháp biến đổi phù hợp với bản chất của từng đặc trưng.

### 5. Tokenization

Các giá trị sau khi biến đổi được rời rạc hóa thành token.

Kết quả cuối cùng là một ma trận:

```text
Number of Network Flows × Number of Features
```

Trong đó mỗi dòng biểu diễn một network flow và mỗi cột tương ứng với một feature token.

---

## 🤖 Deep Learning Model

Mô hình trung tâm của đồ án là **Deep_MHA_Tabular**, được thiết kế dựa trên Transformer và tối ưu cho dữ liệu dạng bảng.

### Feature Embedding

Mỗi token được biểu diễn thông qua hai thành phần:

```text
PLE Value Embedding       → 192 dimensions
Feature Identity Embedding → 64 dimensions
                         ────────────────
Total Embedding           → 256 dimensions
```

PLE giúp bảo toàn thứ tự và quan hệ về độ lớn của các giá trị số, trong khi Feature Identity Embedding giúp mô hình phân biệt ý nghĩa của từng feature.

### Parallel Multi-Branch Attention

Thay vì sử dụng một MHA duy nhất, mô hình triển khai nhiều nhánh Attention độc lập:

```text
                 Input Tokens
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Branch 1    Branch 2    Branch 3    Branch 4
          │           │           │           │
         MHA         MHA         MHA         MHA
          │           │           │           │
          └───────────┴───────────┴───────────┘
                          │
                    Aggregation
                          │
                          ▼
                   Transformer Block
```

Mỗi nhánh học một không gian quan hệ khác nhau giữa các đặc trưng. Kết quả từ các nhánh được tổng hợp để tạo ra biểu diễn ổn định và giàu thông tin hơn.

### Multi-view Pooling

Sau các Transformer blocks, mô hình sử dụng ba phương pháp pooling:

* Attention Pooling
* Max Pooling
* Mean Pooling

Các biểu diễn được kết hợp trước khi đưa vào bộ phân loại MLP.

---

## ⚙️ Training Configuration

| Parameter                   |           Value |
| --------------------------- | --------------: |
| Value Embedding Dimension   |             192 |
| Feature Embedding Dimension |              64 |
| Embedding Dimension         |             256 |
| PLE Segments                |             192 |
| Transformer Layers          |               4 |
| Attention Branches          |               4 |
| Heads / Branch              |               4 |
| Total Attention Heads       |              16 |
| MLP Dropout                 |            0.15 |
| Batch Size                  |             256 |
| Epochs                      |              50 |
| Early Stopping Patience     |              12 |
| Learning Rate               |            1e-3 |
| Weight Decay                |            1e-2 |
| Optimizer                   |           AdamW |
| Scheduler                   |      OneCycleLR |
| Focal Loss Gamma            |             1.5 |
| Label Smoothing             |            0.05 |
| Gradient Clipping           |             1.0 |
| Train / Validation / Test   | 70% / 15% / 15% |
| Random Seed                 |              42 |

Các cấu hình trên được sử dụng trong quá trình huấn luyện và theo dõi bằng MLflow.

---

## 📊 Dataset

Đồ án sử dụng dataset:

**BCCC-cPacket-Cloud-DDoS-2024**

Dataset chứa lưu lượng network flow được tạo trong môi trường cloud mô phỏng mạng doanh nghiệp, với:

* Hơn **300 network-flow features**
* 17 kịch bản tấn công DDoS dựa trên TCP
* 8 hoạt động benign
* Sau preprocessing: **18 classes**
* Tổng cộng khoảng **641,668 network flows**

Phân bố dữ liệu có mức độ mất cân bằng lớn:

| Class                     | Number of Flows |    Ratio |
| ------------------------- | --------------: | -------: |
| Benign                    |         413,199 |   64.40% |
| Attack-TCP-BYPass-V1      |         138,368 |   21.57% |
| 16 attack classes còn lại |          90,101 |   14.04% |
| **Total**                 |     **641,668** | **100%** |

Sự mất cân bằng này là một trong những thách thức chính của bài toán và là lý do mô hình sử dụng **Focal Loss** và đánh giá bằng **Macro-F1**.

---

## 📈 Experimental Results

Mô hình được đánh giá trên hai nhiệm vụ:

### Binary Classification

Phân biệt:

```text
Benign
   vs
Attack
```

Kết quả:

| Metric   |    Score |
| -------- | -------: |
| Accuracy | **0.98** |
| F1-Score | **0.98** |

Kết quả cho thấy mô hình có khả năng phát hiện lưu lượng tấn công ở mức cao.

### Multi-class Classification

Phân loại **18 lớp** gồm 1 lớp Benign và 17 loại tấn công.

| Metric      |    Score |
| ----------- | -------: |
| Accuracy    | **0.90** |
| Weighted-F1 | **0.90** |
| Macro-F1    | **0.49** |

Weighted-F1 cao trong khi Macro-F1 thấp phản ánh rõ sự mất cân bằng giữa các lớp: mô hình hoạt động tốt trên các lớp có nhiều mẫu nhưng vẫn gặp khó khăn với các lớp tấn công hiếm.

---

## 🧪 Training Pipeline

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Remove Leakage Features
     │
     ▼
Hybrid Feature Selection
     │
     ▼
Quantile Profiling
     │
     ▼
Feature Grouping
     │
     ▼
Scale & Tokenization
     │
     ▼
PLE + Feature Embedding
     │
     ▼
Parallel Multi-Branch Attention
     │
     ▼
Multi-view Pooling
     │
     ▼
MLP Classifier
     │
     ▼
Focal Loss
     │
     ▼
AdamW + OneCycleLR
     │
     ▼
Best Model / Checkpoint
```

Dữ liệu được chia theo tỷ lệ **70% training / 15% validation / 15% testing** bằng stratified split để duy trì phân phối lớp, đặc biệt đối với các lớp tấn công hiếm.

---

## 📁 Project Structure

Cấu trúc thư mục có thể tổ chức theo pipeline của đồ án:

```text
DDoS-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── preprocessing/
│   ├── all_feature.py
│   └── ...
│
├── models/
│   ├── Deep_MHA_Tabular.py
│   └── ...
│
├── training/
│   └── start.py
│
├── evaluation/
│   ├── confusion_matrix.py
│   └── ...
│
├── mlruns/
│
├── requirements.txt
├── README.md
└── report/
    └── Final_Report.pdf
```

> Tên file/thư mục có thể thay đổi tùy theo source code thực tế. Báo cáo xác nhận logic huấn luyện được hiện thực trong `start.py` và phần Hybrid Feature Selection nằm trong `all_feature.py`.

---

## 🚀 Installation

### 1. Clone repository

```bash
git clone <repository-url>
cd DDoS-Detection
```

### 2. Tạo virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dataset

Đặt dataset BCCC-cPacket-Cloud-DDoS-2024 vào thư mục dữ liệu theo cấu trúc của project.

```text
data/
└── raw/
    └── BCCC-cPacket-Cloud-DDoS-2024/
```

### 5. Chạy preprocessing

```bash
python preprocessing/all_feature.py
```

### 6. Huấn luyện mô hình

```bash
python training/start.py
```

---

## 📊 Experiment Tracking

Quá trình huấn luyện được theo dõi bằng **MLflow**, bao gồm:

* Training Loss
* Validation Loss
* Accuracy
* Weighted-F1
* Macro-F1
* Hyperparameters
* Model checkpoint

Điều này hỗ trợ theo dõi và tái lập các thí nghiệm trong quá trình phát triển mô hình.

---

## ⚠️ Limitations

Mặc dù mô hình đạt kết quả tốt trong bài toán phát hiện nhị phân, đồ án vẫn có một số hạn chế:

* **Macro-F1 chỉ khoảng 0.49**, cho thấy hiệu năng trên các lớp tấn công hiếm còn hạn chế.
* Một số biến thể TCP có đặc trưng network flow gần như tương đồng, dẫn tới hiện tượng **cross-confusion**.
* Thực nghiệm hiện tại chủ yếu dựa trên một dataset: **BCCC-cPacket-Cloud-DDoS-2024**.
* Chưa đánh giá đầy đủ khả năng tổng quát hóa sang các dataset và loại DDoS khác.
* Chưa đánh giá latency và throughput trong môi trường **real-time / online streaming**.

---

## 🔮 Future Work

Các hướng phát triển có thể tập trung vào:

* Mở rộng thực nghiệm trên nhiều dataset DDoS khác nhau.
* Bổ sung các loại tấn công ngoài TCP-based DDoS.
* Cải thiện khả năng nhận diện các lớp attack có số lượng mẫu thấp.
* Nghiên cứu các kỹ thuật xử lý **class imbalance** nâng cao.
* Tối ưu mô hình cho inference trên môi trường real-time.
* Đánh giá latency, throughput và tài nguyên sử dụng.
* Triển khai mô hình vào hệ thống IDS thực tế.
* Nghiên cứu khả năng giải thích quyết định của mô hình bằng các phương pháp XAI như SHAP/LIME.

---

## 🎯 Conclusion

Đồ án xây dựng một hướng tiếp cận Deep Learning chuyên biệt cho dữ liệu network-flow dạng bảng bằng cách kết hợp **Feature-wise Tokenization** và **Parallel Multi-Branch Attention**.

Kết quả thực nghiệm cho thấy mô hình đạt **0.98 Accuracy/F1 trong bài toán phát hiện Benign/Attack** và **0.90 Accuracy, 0.90 Weighted-F1 trong bài toán phân loại 18 lớp**. Tuy nhiên, sự mất cân bằng dữ liệu vẫn ảnh hưởng đáng kể đến khả năng nhận diện các lớp tấn công hiếm, thể hiện qua Macro-F1 khoảng 0.49.

Đây là nền tảng để tiếp tục phát triển mô hình thành một hệ thống **AI-based Intrusion Detection System** có khả năng hoạt động trong môi trường mạng thực tế.

---

## 👥 Authors

**Nguyễn Huỳnh Nhân**
MSSV: 23521080

**Võ Hoài Nam**
MSSV: 23520990

**Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM**

---

## 📄 Documentation

Báo cáo đầy đủ của đồ án:

`[NT114.Q21.ANTT]-Final Report-23521080.pdf`

**Project:** DDoS Detection using Deep Learning
**Architecture:** Feature-wise Tokenization + Parallel Multi-Branch Attention
**Framework:** PyTorch
**Experiment Tracking:** MLflow
**Year:** 2026
