# Xây dựng hệ thống phát hiện và phân loại phương tiện giao thông trong ảnh

## Giới thiệu

Đây là đồ án cuối kỳ môn **Nhập môn Thị giác máy tính** tại Trường Đại học Công nghệ Thông tin - Đại học Quốc gia Thành phố Hồ Chí Minh.

Mục tiêu của đề tài là xây dựng và đánh giá các phương pháp phát hiện đối tượng (Object Detection) nhằm nhận diện và phân loại các phương tiện giao thông trong ảnh, bao gồm:

- Car
- Motorbike
- Truck
- Bus

Đề tài tiến hành so sánh hiệu năng của bốn phương pháp:

- HOG + SVM
- SSD (Single Shot MultiBox Detector)
- Faster R-CNN
- YOLOv8

---

## Tập dữ liệu

Dataset sử dụng trong đề tài:

**Vehicle Detection Dataset**

Nguồn:  
https://universe.roboflow.com/skripsi-1qzlz/vehicle-detection-bvxpr

Đặc điểm:

- Khoảng 1000 ảnh giao thông thực tế.
- Thu thập từ camera giao thông công cộng tại Yogyakarta, Indonesia.
- Định dạng nhãn YOLO.
- 4 lớp đối tượng:
  - Car
  - Motorbike
  - Truck
  - Bus

---

## Các phương pháp thực nghiệm

### 1. HOG + SVM

Phương pháp học máy truyền thống:

- Trích xuất đặc trưng bằng Histogram of Oriented Gradients (HOG).
- Phân loại bằng Support Vector Machine (SVM).
- Sử dụng Sliding Window và Non-Maximum Suppression (NMS) để phát hiện đối tượng.

### 2. SSD

Mô hình Object Detection một giai đoạn:

- Backbone VGG16.
- Multi-scale Feature Maps.
- Default Boxes (Anchor Boxes).
- Tốc độ suy luận nhanh, phù hợp thời gian thực.

### 3. Faster R-CNN

Mô hình Object Detection hai giai đoạn:

- CNN Backbone.
- Region Proposal Network (RPN).
- ROI Pooling.
- Classification và Bounding Box Regression.

### 4. YOLOv8

Mô hình được lựa chọn làm phương pháp chính:

- Backbone C2f + SPPF.
- Anchor-free Detection Head.
- Hỗ trợ phát hiện đa tỉ lệ.
- Tốc độ suy luận cao và độ chính xác vượt trội.

---

## Kết quả thực nghiệm

| Phương pháp  | Precision | Recall | mAP50  | mAP50-95 | FPS    |
| ------------ | --------- | ------ | ------ | -------- | ------ |
| HOG + SVM    | 4.30%     | 0.62%  | 0.12%  | 0.03%    | 5.00   |
| SSD          | 38.70%    | 38.60% | 79.20% | 54.60%   | 26.00  |
| Faster R-CNN | 39.62%    | 39.81% | 91.79% | 67.43%   | 12.21  |
| YOLOv8       | 94.56%    | 90.72% | 95.38% | 76.19%   | 142.68 |

### Nhận xét

- HOG + SVM có hiệu năng thấp khi áp dụng cho bài toán giao thông thực tế.
- SSD đạt tốc độ xử lý tốt nhưng độ chính xác còn hạn chế.
- Faster R-CNN cho độ chính xác cao nhưng tốc độ suy luận thấp.
- YOLOv8 đạt sự cân bằng tốt nhất giữa độ chính xác và tốc độ xử lý.

---

## Cấu trúc thư mục

```text
CS231-VehicleCounting/
├── data/
├── models/
│   ├── best_FasterRCNN.pth
│   ├── best_hog_svm.pkl
│   ├── best_SSD.pth
│   └── yolov8.pt
├── notebooks/
│   ├── EDA_main.ipynb
│   ├── deployment.ipynb
│   ├── FasterRCNN/
│   │   └── FasterRCNN_main.ipynb
│   ├── HOG_SVM/
│   │   ├── HOG_AND_SVM .ipynb
│   │   └── hog_svm_vehicle_detection.ipynb
│   ├── SSD/
│   │   └── SSD_main.ipynb
│   └── YOLOv8/
│       ├── yolov8_aug_main.ipynb
│       └── yolov8_no_aug_main.ipynb
├── .gitattributes
├── .gitignore
└── README.md
```

---

## Cài đặt môi trường

Tạo môi trường Python:

```bash
python -m venv venv
```

Kích hoạt môi trường:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## Thành viên thực hiện

- Đoàn Hữu Gia Bình - 24520192
- Phan Công Thiện - 24521665
- Nguyễn Tấn Phát - 24521305
- Nguyễn Trường Nguyên - 24521200

Giảng viên hướng dẫn:

- ThS. Mai Tiến Dũng

---

## Kết luận

Đề tài đã xây dựng thành công hệ thống phát hiện và phân loại phương tiện giao thông dựa trên các kỹ thuật Thị giác máy tính hiện đại. Kết quả thực nghiệm cho thấy YOLOv8 là phương pháp phù hợp nhất cho bài toán nhờ đạt độ chính xác cao và tốc độ suy luận nhanh, đáp ứng tốt các yêu cầu của hệ thống giám sát giao thông thời gian thực.
