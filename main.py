import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Đường dẫn đến thư mục chứa ảnh và thư mục lưu ảnh cắt
img_dir = "./images/val"
output_dir = "./labels/val"

# Kiểm tra nếu không có ảnh
if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Thư mục {img_dir} không tồn tại.")

# Kiểm tra nếu không có ảnh trong thư mục
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not img_files:
    raise FileNotFoundError("Không có tệp ảnh nào trong thư mục.")

# Đảm bảo thư mục đầu ra tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hàm hiển thị ảnh cắt
def show_cropped_image(cropped_img, title=""):
    plt.imshow(cropped_img, cmap='gray')  # Hiển thị ảnh ở chế độ thang độ xám
    plt.title(title, fontsize=10)
    plt.axis('off')  # Ẩn trục tọa độ
    plt.show()

# Danh sách để lưu các tên nhãn, sử dụng set để tránh trùng lặp
names_set = set()
# Từ điển để lưu class_id cho từng nhãn
class_id_map = {}

# Hàm xử lý từng tấm ảnh
def process_image(img_path):
    # Đọc ảnh gốc
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc được ảnh từ đường dẫn: {img_path}")
    
    # Xử lý ảnh (như trong đoạn mã ban đầu)
    thresh_img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    close_img1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))
    dilate_img1 = cv2.dilate(close_img1, np.ones((2, 2), np.uint8), iterations=1)
    gauss_img1 = cv2.GaussianBlur(dilate_img1, (1, 1), 0)

    # Vùng bounding box cố định (nếu áp dụng cho tất cả ảnh)
    rectangles = [
        (30, 7, 50, 49),
        (45, 7, 75, 49),
        (68, 7, 98, 49),
        (90, 7, 120, 49),
        (112, 7, 145, 49),
    ]
    
    # Nhãn từ tên ảnh
    labels = list(os.path.splitext(os.path.basename(img_path))[0])
    if len(labels) != len(rectangles):
        raise ValueError(f"Số lượng nhãn không khớp với số vùng đã chỉ định trong ảnh {img_path}.")
    
    yolo_data = []  # Dữ liệu nhãn theo định dạng YOLO
    
    # Cắt ảnh và hiển thị
    for idx, (x1, y1, x2, y2) in enumerate(rectangles):
        # Cắt ảnh từ vùng chỉ định
        cropped_img = gauss_img1[y1:y2, x1:x2]
        
        # Hiển thị ảnh cắt
        # show_cropped_image(cropped_img, title=f"Ảnh: {os.path.basename(img_path)}, Class ID: {labels[idx]}")
        
        # Tính toán các thông số cho YOLO
        img_height, img_width = img.shape
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        box_width = (x2 - x1) / img_width
        box_height = (y2 - y1) / img_height

        # Kiểm tra nếu nhãn đã tồn tại trong từ điển
        if labels[idx] not in class_id_map:
            # Nếu chưa tồn tại, gán class_id mới
            class_id_map[labels[idx]] = len(class_id_map)
        
        # Lấy class_id cho nhãn hiện tại
        class_id = class_id_map[labels[idx]]

        yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # Thêm tên nhãn vào set để tránh trùng lặp
        names_set.add(labels[idx])

    # Lưu nhãn vào tệp .txt
    label_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
    with open(label_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_data))

# Lặp qua từng ảnh trong thư mục và xử lý
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    try:
        process_image(img_path)
        print(f"Đã xử lý và lưu ảnh: {img_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

# Chuyển set thành list để dễ dàng hiển thị và tránh trùng lặp
names = list(names_set)

# Hiển thị danh sách tên nhãn
print("Danh sách tên nhãn của các ảnh cắt:", names)
