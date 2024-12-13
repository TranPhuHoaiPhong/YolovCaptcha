import cv2
from ultralytics import YOLO

# Đường dẫn đến mô hình đã huấn luyện
model_path = r"C:\Users\MSI\runs\detect\train29\weights\best.pt"
# Đường dẫn đến hình ảnh cần dự đoán
image_path = r"C:\MainPlace\Work\PythonApp\Mô hình Yolov (Đồ án)\2b827.png"

# Tải mô hình YOLOv8
model = YOLO(model_path)

# Thực hiện dự đoán
results = model(image_path)  # Dự đoán trực tiếp trên hình ảnh

# In thông tin các dự đoán
print("Các kết quả nhận diện:")
for box in results[0].boxes:
    cls = int(box.cls)  # Lớp dự đoán (dạng số)
    confidence = box.conf  # Độ tin cậy
    number = model.names[cls]  # Tên lớp tương ứng
    print(f"Lớp: {number}, Độ tin cậy: {confidence:.2f}")

# Hiển thị kết quả hình ảnh
annotated_img = results[0].plot()  # Vẽ kết quả nhận diện trên ảnh
cv2.imshow("Kết quả", annotated_img)  # Hiển thị hình ảnh
cv2.waitKey(0)  # Đợi nhấn phím để đóng cửa sổ
cv2.destroyAllWindows()  # Đóng tất cả cửa sổ
