import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Đường dẫn đến thư mục chứa ảnh và thư mục lưu nhãn
img_dir = "./images/val"
output_dir = "./labels/test"
a = []

# Kiểm tra nếu thư mục ảnh không tồn tại
if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Thư mục {img_dir} không tồn tại.")

img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])



if not img_files:
    raise FileNotFoundError("Không có tệp ảnh nào trong thư mục.")

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Danh sách lưu tên nhãn theo thứ tự xuất hiện
names_list = []
class_id_map = {}

def show_cropped_image(cropped_img, title=""):
    plt.imshow(cropped_img, cmap='gray')  # Hiển thị ảnh ở chế độ thang độ xám
    plt.title(title, fontsize=10)
    plt.axis('off')  # Ẩn trục tọa độ
    plt.show()

# Hàm xử lý từng tấm ảnh
def process_image(img_path):
    # Đọc ảnh gốc
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc được ảnh từ đường dẫn: {img_path}")
    
    # Xử lý ảnh
    thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    close_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))
    dilate_img = cv2.dilate(close_img, np.ones((2, 2), np.uint8), iterations=1)
    processed_img = cv2.GaussianBlur(dilate_img, (1, 1), 0)

    # Vùng bounding box cố định
    rectangles = [
        (30, 7, 50, 49),
        (50, 7, 70, 49),
        (70, 7, 90, 49),
        (90, 7, 110, 49),
        (110, 7, 150, 49),
    ]
    
    # Lấy nhãn từ tên tệp ảnh
    labels = list(os.path.splitext(os.path.basename(img_path))[0])
  
    # Thêm nhãn không trùng lặp vào danh sách a
    for i in labels:
        if i not in a:
            a.append(i)

    if len(labels) != len(rectangles):
        raise ValueError(f"Số lượng nhãn không khớp với số vùng trong ảnh {img_path}.")

    yolo_data = []  # Dữ liệu nhãn theo định dạng YOLO
    unique_labels = set()  # Đảm bảo nhãn không bị trùng lặp trong mỗi ảnh

    # Xử lý từng bounding box
    for idx, (x1, y1, x2, y2) in enumerate(rectangles):
        # Tính toán các thông số cho YOLO
        img_height, img_width = img.shape
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        box_width = (x2 - x1) / img_width
        box_height = (y2 - y1) / img_height

        # Gán class_id cho nhãn, loại bỏ nhãn trùng lặp và giữ thứ tự
        for label in labels:
            if label not in unique_labels:
                unique_labels.add(label)
                if label not in class_id_map:
                    # Thêm nhãn mới vào danh sách và gán class_id
                    class_id_map[label] = len(names_list)
                    names_list.append(label)  # Giữ thứ tự xuất hiện

        # Ghi nhận nhãn của từng bounding box
        for label in unique_labels:
            class_id = class_id_map[label]
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Lưu dữ liệu nhãn vào tệp .txt
    label_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
    with open(label_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_data))

# Lặp qua từng ảnh và xử lý
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    try:
        process_image(img_path)
        print(f"Đã xử lý và lưu nhãn cho ảnh: {img_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

# Lưu danh sách nhãn vào classes.txt
classes_file_path = os.path.join(output_dir, "classes.txt")
with open(classes_file_path, "w", encoding="utf-8") as f:
    for name in names_list:
        f.write(name + "\n")

print(f"Danh sách nhãn không trùng lặp: {a}")
