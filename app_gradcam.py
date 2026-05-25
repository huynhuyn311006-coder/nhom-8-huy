import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
from tensorflow.keras.preprocessing import image

# =========================
# CẤU HÌNH CHUNG
# =========================

MODEL_PATH = "thyroid_model_v3.keras"
IMG_SIZE = (224, 224)
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    compile=False
)


# =========================
# HÀM DIỄN GIẢI KẾT QUẢ
# =========================

def diagnose_result(prediction_value, attention_percent):
    """
    prediction_value gần 0: benign
    prediction_value gần 1: malignant
    attention_percent: tỷ lệ vùng Grad-CAM chú ý mạnh
    """

    malignant_score = prediction_value * 100
    benign_score = (1 - prediction_value) * 100

    # Nhận xét vùng Grad-CAM
    if attention_percent < 3:
        attention_comment = (
            "Vùng chú ý của mô hình nhỏ và khu trú. "
            "Mô hình chỉ tập trung vào một vùng hẹp trên ảnh."
        )
    elif attention_percent < 10:
        attention_comment = (
            "Vùng chú ý của mô hình ở mức vừa, tập trung tương đối rõ vào vùng nghi ngờ."
        )
    elif attention_percent < 20:
        attention_comment = (
            "Vùng chú ý của mô hình khá rộng. Cần quan sát kỹ vùng được đánh dấu."
        )
    else:
        attention_comment = (
            "Vùng chú ý của mô hình lan rộng. Cần thận trọng vì mô hình có thể bị ảnh hưởng bởi nhiều vùng ảnh."
        )

    # Phân loại nguy cơ
    if prediction_value < 0.10:
        diagnosis = "Không phát hiện dấu hiệu bất thường rõ"
        final_class = "Benign"
        risk_level = "Nguy cơ rất thấp"
        confidence = benign_score
        medical_comment = (
            "Mô hình đánh giá ảnh có xu hướng lành tính rõ. "
            "Chưa phát hiện đặc điểm mạnh gợi ý ác tính trên ảnh đầu vào."
        )
        recommendation = (
            "Khuyến nghị: Theo dõi định kỳ. Nếu không có triệu chứng bất thường, "
            "có thể tiếp tục kiểm tra theo lịch khám thông thường."
        )

    elif prediction_value < 0.30:
        diagnosis = "Khả năng lành tính cao"
        final_class = "Benign"
        risk_level = "Nguy cơ thấp"
        confidence = benign_score
        medical_comment = (
            "Mô hình nghiêng về nhóm lành tính. "
            "Tuy nhiên vẫn nên kết hợp với thông tin lâm sàng và mô tả siêu âm."
        )
        recommendation = (
            "Khuyến nghị: Theo dõi định kỳ bằng siêu âm. "
            "Nếu nốt tăng kích thước hoặc có triệu chứng vùng cổ, nên khám chuyên khoa."
        )

    elif prediction_value < 0.45:
        diagnosis = "Nghiêng về lành tính nhưng cần theo dõi"
        final_class = "Benign"
        risk_level = "Nguy cơ thấp đến trung bình"
        confidence = benign_score
        medical_comment = (
            "Mô hình vẫn nghiêng về lành tính nhưng điểm dự đoán chưa cách xa ngưỡng 0.5. "
            "Trường hợp này nên được theo dõi kỹ hơn."
        )
        recommendation = (
            "Khuyến nghị: Nên được bác sĩ đọc lại ảnh. "
            "Có thể cần đánh giá thêm theo TI-RADS nếu có đủ đặc điểm siêu âm."
        )

    elif prediction_value < 0.55:
        diagnosis = "Kết quả chưa rõ ràng"
        final_class = "Không chắc chắn"
        risk_level = "Nguy cơ trung gian"
        confidence = max(benign_score, malignant_score)
        medical_comment = (
            "Điểm dự đoán nằm gần ngưỡng phân loại. "
            "Mô hình chưa đủ tự tin để kết luận rõ lành tính hay nghi ngờ ác tính."
        )
        recommendation = (
            "Khuyến nghị: Không nên dựa riêng vào AI. "
            "Cần bác sĩ chuyên khoa đánh giá lại ảnh, kết hợp triệu chứng và xét nghiệm nếu cần."
        )

    elif prediction_value < 0.70:
        diagnosis = "Có dấu hiệu nghi ngờ bất thường"
        final_class = "Malignant"
        risk_level = "Nguy cơ trung bình"
        confidence = malignant_score
        medical_comment = (
            "Mô hình phát hiện một số đặc điểm ảnh có xu hướng thuộc nhóm bất thường. "
            "Mức điểm chưa quá cao nhưng cần lưu ý."
        )
        recommendation = (
            "Khuyến nghị: Nên khám chuyên khoa Nội tiết hoặc Chẩn đoán hình ảnh. "
            "Có thể cần siêu âm lại hoặc theo dõi kích thước nốt."
        )

    elif prediction_value < 0.85:
        diagnosis = "Nghi ngờ ác tính mức cao"
        final_class = "Malignant"
        risk_level = "Nguy cơ cao"
        confidence = malignant_score
        medical_comment = (
            "Mô hình nghiêng rõ về nhóm ác tính hoặc bất thường. "
            "Vùng Grad-CAM nên được xem là vùng gợi ý để kiểm tra kỹ hơn."
        )
        recommendation = (
            "Khuyến nghị: Cần khám chuyên khoa sớm. "
            "Bác sĩ có thể cân nhắc siêu âm chuyên sâu, phân loại TI-RADS hoặc chọc hút tế bào nếu phù hợp."
        )

    elif prediction_value < 0.95:
        diagnosis = "Nghi ngờ ác tính rất cao"
        final_class = "Malignant"
        risk_level = "Nguy cơ rất cao"
        confidence = malignant_score
        medical_comment = (
            "Mô hình cho điểm ác tính rất cao. "
            "Kết quả này cần được ưu tiên kiểm tra lại bởi bác sĩ chuyên khoa."
        )
        recommendation = (
            "Khuyến nghị: Nên đi khám chuyên khoa Nội tiết, Ung bướu hoặc Chẩn đoán hình ảnh sớm. "
            "Có thể cần làm thêm xét nghiệm hoặc chọc hút tế bào nếu bác sĩ chỉ định."
        )

    else:
        diagnosis = "Cảnh báo nguy cơ ác tính đặc biệt cao"
        final_class = "Malignant"
        risk_level = "Cảnh báo rất cao"
        confidence = malignant_score
        medical_comment = (
            "Mô hình gần như chắc chắn xếp ảnh vào nhóm malignant. "
            "Tuy nhiên đây vẫn là kết quả AI, không thay thế kết luận y khoa."
        )
        recommendation = (
            "Khuyến nghị: Cần được bác sĩ chuyên khoa đánh giá trực tiếp. "
            "Không nên trì hoãn việc thăm khám nếu có triệu chứng hoặc nốt tuyến giáp bất thường."
        )

    return {
        "final_class": final_class,
        "diagnosis": diagnosis,
        "risk_level": risk_level,
        "confidence": confidence,
        "malignant_score": malignant_score,
        "benign_score": benign_score,
        "attention_percent": attention_percent,
        "attention_comment": attention_comment,
        "medical_comment": medical_comment,
        "recommendation": recommendation
    }


# =========================
# HÀM TẠO GRAD-CAM
# =========================

def make_gradcam(file_path):
    # Load ảnh
    img = image.load_img(file_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # QUAN TRỌNG:
    # Không chia 255 ở đây vì trong mô hình train.py đã có lớp Rescaling(1./255)
    prediction_value = model.predict(img_array)[0][0]

    # Lấy base model MobileNetV2 trong mô hình Sequential
    base_model = model.layers[2]

    # Tìm lớp Conv2D cuối cùng
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise ValueError("Không tìm thấy lớp Conv2D cuối cùng trong mô hình.")

    # Tạo mô hình Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    # QUAN TRỌNG:
    # Base model nhận ảnh đã scale 0-1, nên chỉ dùng base_input cho Grad-CAM
    base_input = img_array / 255.0

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(base_input)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    # Đọc ảnh gốc
    original_img = cv2.imread(file_path)

    if original_img is None:
        raise ValueError("Không đọc được ảnh. Vui lòng chọn ảnh .jpg, .jpeg hoặc .png hợp lệ.")

    original_img = cv2.resize(original_img, IMG_SIZE)

    # Resize heatmap
    heatmap = cv2.resize(heatmap, IMG_SIZE)

    # Tạo mask vùng chú ý mạnh
    threshold = 0.5
    mask = heatmap > threshold
    mask = np.uint8(mask) * 255

    # Tính tỷ lệ vùng chú ý
    attention_percent = (np.sum(mask > 0) / mask.size) * 100

    # Diễn giải kết quả
    diagnosis_info = diagnose_result(prediction_value, attention_percent)

    # Tạo heatmap màu
    heatmap_color = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

    # Ghép heatmap lên ảnh gốc
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # Tìm vùng contour
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Khoanh vùng nghi ngờ
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                superimposed_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Ghi nhãn ngắn lên ảnh
    short_text = f"{diagnosis_info['final_class']} - {diagnosis_info['risk_level']}"
    cv2.putText(
        superimposed_img,
        short_text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1
    )

    # Lưu ảnh và TXT
    time_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_image_path = os.path.join(RESULT_DIR, f"gradcam_result_{time_name}.jpg")
    save_txt_path = os.path.join(RESULT_DIR, f"diagnosis_result_{time_name}.txt")

    cv2.imwrite(save_image_path, superimposed_img)

    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ AI HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM TUYẾN GIÁP\n")
        f.write("=================================================\n")
        f.write(f"Tên ảnh: {os.path.basename(file_path)}\n\n")

        f.write("1. KẾT QUẢ PHÂN LOẠI\n")
        f.write(f"- Kết luận AI: {diagnosis_info['diagnosis']}\n")
        f.write(f"- Nhóm dự đoán: {diagnosis_info['final_class']}\n")
        f.write(f"- Mức nguy cơ: {diagnosis_info['risk_level']}\n")
        f.write(f"- Điểm benign: {diagnosis_info['benign_score']:.2f}%\n")
        f.write(f"- Điểm malignant: {diagnosis_info['malignant_score']:.2f}%\n")
        f.write(f"- Độ tin cậy: {diagnosis_info['confidence']:.2f}%\n\n")

        f.write("2. NHẬN XÉT VÙNG NGHI NGỜ\n")
        f.write(f"- Tỷ lệ vùng Grad-CAM chú ý mạnh: {diagnosis_info['attention_percent']:.2f}%\n")
        f.write(f"- Nhận xét: {diagnosis_info['attention_comment']}\n\n")

        f.write("3. DIỄN GIẢI SƠ BỘ\n")
        f.write(f"{diagnosis_info['medical_comment']}\n\n")

        f.write("4. KHUYẾN NGHỊ\n")
        f.write(f"{diagnosis_info['recommendation']}\n\n")

        f.write("LƯU Ý\n")
        f.write(
            "Kết quả này chỉ phục vụ nghiên cứu và học tập, "
            "không thay thế chẩn đoán của bác sĩ hoặc chuyên gia y tế.\n"
        )

    return {
        "prediction_value": prediction_value,
        "diagnosis_info": diagnosis_info,
        "gradcam_img": superimposed_img,
        "save_image_path": save_image_path,
        "save_txt_path": save_txt_path
    }


# =========================
# GIAO DIỆN TKINTER CÓ THANH CUỘN
# =========================

root = tk.Tk()
root.title("AI hỗ trợ phát hiện bất thường tuyến giáp")
root.geometry("1050x700")

# Canvas chính để có thanh cuộn
main_canvas = tk.Canvas(root)
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

main_canvas.configure(yscrollcommand=scrollbar.set)

main_frame = tk.Frame(main_canvas)
main_canvas.create_window((0, 0), window=main_frame, anchor="nw")


def update_scroll_region(event):
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))


main_frame.bind("<Configure>", update_scroll_region)


def mouse_wheel(event):
    main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


main_canvas.bind_all("<MouseWheel>", mouse_wheel)

title_label = tk.Label(
    main_frame,
    text="AI hỗ trợ phát hiện bất thường tuyến giáp trên ảnh siêu âm",
    font=("Arial", 16, "bold")
)
title_label.pack(pady=10)

result_label = tk.Label(
    main_frame,
    text="Chưa chọn ảnh",
    font=("Arial", 11),
    justify="left",
    wraplength=980
)
result_label.pack(pady=10)

frame = tk.Frame(main_frame)
frame.pack(pady=10)

original_title = tk.Label(
    frame,
    text="Ảnh gốc",
    font=("Arial", 12, "bold")
)
original_title.grid(row=0, column=0, padx=20)

gradcam_title = tk.Label(
    frame,
    text="Ảnh khoanh vùng nghi ngờ bằng Grad-CAM",
    font=("Arial", 12, "bold")
)
gradcam_title.grid(row=0, column=1, padx=20)

original_image_label = tk.Label(frame)
original_image_label.grid(row=1, column=0, padx=20, pady=10)

gradcam_image_label = tk.Label(frame)
gradcam_image_label.grid(row=1, column=1, padx=20, pady=10)

save_label = tk.Label(
    main_frame,
    text="",
    font=("Arial", 10),
    fg="blue",
    justify="left",
    wraplength=980
)
save_label.pack(pady=5)


def choose_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh siêu âm tuyến giáp",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png")
        ]
    )

    if not file_path:
        return

    try:
        # Hiển thị ảnh gốc
        img_original = Image.open(file_path)
        img_original = img_original.resize((360, 360))
        img_original_tk = ImageTk.PhotoImage(img_original)

        original_image_label.config(image=img_original_tk)
        original_image_label.image = img_original_tk

        # Phân tích AI
        result = make_gradcam(file_path)
        info = result["diagnosis_info"]

        # Hiển thị kết quả
        result_label.config(
            text=(
                f"KẾT QUẢ AI\n"
                f"- Kết luận: {info['diagnosis']}\n"
                f"- Nhóm dự đoán: {info['final_class']}\n"
                f"- Mức nguy cơ: {info['risk_level']}\n"
                f"- Điểm benign: {info['benign_score']:.2f}%\n"
                f"- Điểm malignant: {info['malignant_score']:.2f}%\n"
                f"- Độ tin cậy: {info['confidence']:.2f}%\n"
                f"- Vùng Grad-CAM chú ý mạnh: {info['attention_percent']:.2f}% ảnh\n\n"
                f"NHẬN XÉT\n"
                f"{info['medical_comment']}\n"
                f"{info['attention_comment']}\n\n"
                f"KHUYẾN NGHỊ\n"
                f"{info['recommendation']}"
            )
        )

        # Hiển thị ảnh Grad-CAM
        gradcam_rgb = cv2.cvtColor(result["gradcam_img"], cv2.COLOR_BGR2RGB)
        gradcam_pil = Image.fromarray(gradcam_rgb)
        gradcam_pil = gradcam_pil.resize((360, 360))
        gradcam_tk = ImageTk.PhotoImage(gradcam_pil)

        gradcam_image_label.config(image=gradcam_tk)
        gradcam_image_label.image = gradcam_tk

        # Hiển thị file lưu
        save_label.config(
            text=(
                f"Đã lưu ảnh khoanh vùng: {result['save_image_path']}\n"
                f"Đã lưu phiếu kết quả TXT: {result['save_txt_path']}"
            )
        )

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))


choose_button = tk.Button(
    main_frame,
    text="Chọn ảnh siêu âm để AI phân tích",
    font=("Arial", 13),
    command=choose_image
)
choose_button.pack(pady=15)

note_label = tk.Label(
    main_frame,
    text=(
        "Lưu ý: Đây là hệ thống AI hỗ trợ nghiên cứu và học tập. "
        "Kết quả không thay thế chẩn đoán của bác sĩ, không dùng làm căn cứ điều trị."
    ),
    font=("Arial", 10),
    fg="red",
    wraplength=950
)
note_label.pack(pady=10)

root.mainloop()