import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("thyroid_model.h5")

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Phát hiện bất thường tuyến giáp trên ảnh siêu âm")
root.geometry("600x650")

# Tiêu đề
title_label = tk.Label(
    root,
    text="Phát hiện bất thường tuyến giáp trên ảnh siêu âm",
    font=("Arial", 16, "bold")
)
title_label.pack(pady=15)

# Khung hiển thị ảnh
image_label = tk.Label(root)
image_label.pack(pady=10)

# Hiển thị kết quả
result_label = tk.Label(
    root,
    text="Chưa chọn ảnh",
    font=("Arial", 14)
)
result_label.pack(pady=10)

# Hàm chọn ảnh và dự đoán
def choose_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh siêu âm tuyến giáp",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png")
        ]
    )

    if not file_path:
        return

    # Hiển thị ảnh trên giao diện
    img_display = Image.open(file_path)
    img_display = img_display.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Tiền xử lý ảnh cho mô hình
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Dự đoán
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        result = f"Kết quả: Malignant - Có dấu hiệu bất thường / ác tính\nGiá trị dự đoán: {prediction:.4f}"
    else:
        result = f"Kết quả: Benign - Lành tính\nGiá trị dự đoán: {prediction:.4f}"

    result_label.config(text=result)

# Nút chọn ảnh
choose_button = tk.Button(
    root,
    text="Chọn ảnh để dự đoán",
    font=("Arial", 13),
    command=choose_image
)
choose_button.pack(pady=20)

# Ghi chú
note_label = tk.Label(
    root,
    text="Lưu ý: Kết quả chỉ phục vụ nghiên cứu, không thay thế chẩn đoán y khoa.",
    font=("Arial", 10),
    fg="red",
    wraplength=500
)
note_label.pack(pady=10)

# Chạy giao diện
root.mainloop()