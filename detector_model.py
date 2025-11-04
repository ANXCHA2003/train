import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
import sys
import os
import json

def load_model_and_labels(run_dir):
    """โหลดโมเดลและ class labels จากโฟลเดอร์ผลลัพธ์ที่ระบุ"""
    model_path = os.path.join(run_dir, 'best_model.h5')
    labels_path = os.path.join(run_dir, 'class_labels.json')

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print(f"Error: ไม่พบไฟล์ 'best_model.h5' หรือ 'class_labels.json' ในโฟลเดอร์ '{run_dir}'")
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, 'r') as f:
        labels_from_file = json.load(f)
    class_labels = {int(k): v for k, v in labels_from_file.items()}
    
    return model, class_labels

# --- 2. ฟังก์ชันสำหรับทำนาย ---
def predict_image(model, class_labels, image_path):
    if not os.path.exists(image_path):
        print(f"Error: ไม่พบไฟล์ที่ '{image_path}'")
        return

    # โหลดและเตรียมรูปภาพ
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # ทำให้เป็น 4 มิติ (batch, height, width, channels)
    img_array /= 255.0 # ปรับค่าสีให้อยู่ในช่วง 0-1 เหมือนตอนฝึก
    
    # ทำนายผล
    prediction = model.predict(img_array)
    
    # แปลผล
    score = prediction[0][0]
    print(f"Prediction Score: {score:.4f}")

    # ใช้ชื่อคลาสจากไฟล์ที่โหลดมาเพื่อแสดงผล
    positive_class_name = class_labels.get(1, "Positive Class") # คลาสที่ผลทำนาย > 0.5
    negative_class_name = class_labels.get(0, "Negative Class") # คลาสที่ผลทำนาย <= 0.5

    if score > 0.5: # 0.5 คือค่ากลาง (threshold)
        print(f"ผลการทำนาย: {positive_class_name}")
    else:
        print(f"ผลการทำนาย: {negative_class_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"วิธีใช้งาน: python {script_name} <path_to_run_folder> <path_to_image>")
        print(f"ตัวอย่าง: python {script_name} runs\\2025-11-04_18-00-00 my_orange.jpg")
    else:
        run_folder = sys.argv[1]
        image_file = sys.argv[2]

        model, class_labels = load_model_and_labels(run_folder)

        if model and class_labels:
            predict_image(model, class_labels, image_file)
