from ultralytics import YOLO

# Tải mô hình YOLOv8 đã huấn luyện sẵn hoặc khởi tạo một mô hình mới
model = YOLO("yolov8n.pt")  # Sử dụng mô hình YOLOv8 nhỏ nhất (yolov8n)

model.train(data="data.yaml", epochs=50)
