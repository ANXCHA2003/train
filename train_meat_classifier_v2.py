import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  # ใช้ EfficientNetB0 แทน V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW  # ใช้ AdamW ที่ดีกว่า Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import glob
import json
import os
from datetime import datetime
import csv

# --- 1. กำหนดค่าพื้นฐาน (ปรับปรุงใหม่) ---
IMAGE_SIZE = (240, 240)  # เพิ่มขนาดรูปภาพ
BATCH_SIZE = 32  # เพิ่ม batch size
TEST_SPLIT = 0.2
RANDOM_STATE = 42

# ใช้ข้อมูลที่ปรับสมดุลแล้ว
DATA_DIR = 'meat database/balanced_train'  # เปลี่ยนเป็นข้อมูลที่ปรับสมดุลแล้ว
INITIAL_EPOCHS = 100  # เพิ่ม epochs
FINE_TUNE_EPOCHS = 50   # เปิด fine-tuning

SUMMARY_FILE = 'runs_summary_v2.csv'

# สร้างโฟลเดอร์สำหรับจัดเก็บผลลัพธ์
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR = os.path.join('runs_v2', timestamp)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
print(f"ผลลัพธ์ของการฝึกสอนครั้งนี้จะถูกเก็บไว้ที่: {RUN_OUTPUT_DIR}")

# --- 2. เตรียมข้อมูลรูปภาพ (ปรับปรุงใหม่) ---
print("\n--- เริ่มกระบวนการเตรียมข้อมูล ---")

# ตรวจสอบว่ามีข้อมูลที่ปรับสมดุลแล้วหรือไม่
if not os.path.exists(DATA_DIR):
    print(f"ไม่พบโฟลเดอร์ {DATA_DIR}")
    print("กรุณารัน improve_dataset.py ก่อน")
    exit(1)

# ค้นหาชื่อคลาส
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
num_classes = len(class_names)
class_to_int = {name: i for i, name in enumerate(class_names)}
int_to_class = {i: name for i, name in enumerate(class_names)}

print(f"พบคลาสทั้งหมด {num_classes} คลาส: {class_names}")

# บันทึก Class Labels
class_labels_path = os.path.join(RUN_OUTPUT_DIR, 'class_labels.json')
with open(class_labels_path, 'w', encoding='utf-8') as f:
    json.dump(int_to_class, f, ensure_ascii=False)

# ค้นหาไฟล์รูปภาพ
image_paths = []
labels = []

for class_name in class_names:
    class_path = os.path.join(DATA_DIR, class_name)
    class_images = glob.glob(os.path.join(class_path, '*.jpg')) + glob.glob(os.path.join(class_path, '*.png'))
    
    for img_path in class_images:
        image_paths.append(img_path)
        labels.append(class_to_int[class_name])

print(f"พบรูปภาพทั้งหมด {len(image_paths)} รูป")

# แสดงจำนวนรูปภาพในแต่ละคลาส
for class_name in class_names:
    count = labels.count(class_to_int[class_name])
    print(f"  {class_name}: {count} รูป")

# แบ่งข้อมูล Train / Validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE, 
    shuffle=True, stratify=labels
)

print(f"จำนวนข้อมูล Train: {len(train_paths)}, Validation: {len(val_paths)}")

# คำนวณ Class Weights เพื่อจัดการ class imbalance
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weight_dict}")

# โหลดและแปลงรูปภาพ
def load_and_preprocess_images(paths, labels, image_size, num_classes):
    images_np = np.zeros((len(paths), *image_size, 3), dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    
    for i, path in enumerate(paths):
        try:
            img = load_img(path, target_size=image_size)
            img_array = img_to_array(img)
            images_np[i] = img_array
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # ใช้รูปภาพสีดำแทน
            images_np[i] = np.zeros((*image_size, 3))
    
    # Normalize to [0, 1] for EfficientNetB0
    images_np = images_np / 255.0
    
    # Convert labels to one-hot encoding
    labels_np = to_categorical(labels_np, num_classes=num_classes)
    
    return images_np, labels_np

print("กำลังโหลดและแปลงรูปภาพ...")
x_train, y_train = load_and_preprocess_images(train_paths, train_labels, IMAGE_SIZE, num_classes)
x_val, y_val = load_and_preprocess_images(val_paths, val_labels, IMAGE_SIZE, num_classes)

# --- 3. Data Augmentation ที่ทันสมัย ---
train_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=0.3,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
validation_data = (x_val, y_val)

# --- 4. สร้างโมเดลที่ทันสมัย ---
print("\n--- สร้างโมเดล ---")

# ใช้ EfficientNetB0 (V2 อาจไม่รองรับในเวอร์ชันนี้)
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(*IMAGE_SIZE, 3)
)

# แช่แข็งโมเดลพื้นฐาน
base_model.trainable = False

# สร้างชั้นบนสุดที่ซับซ้อนขึ้น
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # เพิ่ม Batch Normalization
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)  # เพิ่มขนาด
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. คอมไพล์และฝึกสอน Phase 1 ---
print("\n--- Phase 1: Feature Extraction ---")

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks ที่ทันสมัย
best_model_path = os.path.join(RUN_OUTPUT_DIR, 'best_model_v2.h5')

callbacks = [
    EarlyStopping(
        monitor='val_accuracy', 
        patience=20, 
        verbose=1, 
        mode='max', 
        restore_best_weights=True
    ),
    ModelCheckpoint(
        best_model_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1, 
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=10, 
        verbose=1, 
        min_lr=1e-7
    )
]

# Phase 1: Feature Extraction
history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_data,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# --- 6. Phase 2: Fine-tuning ---
print("\n--- Phase 2: Fine-tuning ---")

# เปิด fine-tuning สำหรับ layer บนสุด
base_model.trainable = True

# แช่แข็ง layer แรกๆ
for layer in base_model.layers[:-30]:  # แช่แข็ง layer แรก 30 ชั้น
    layer.trainable = False

# คอมไพล์ใหม่ด้วย learning rate ที่ต่ำกว่า
model.compile(
    optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning
history2 = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_data,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# --- 7. ประเมินผลและแสดงกราฟ ---
def plot_history_v2(history1, history2, output_dir):
    """พล็อตกราฟแสดงผลการฝึกสอน 2 เฟส"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # รวม history
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    fine_tune_start = len(history1.history['accuracy'])
    
    plt.figure(figsize=(16, 6))
    plt.suptitle('Training History - EfficientNetB0 v2 with Fine-tuning', fontsize=16)
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='royalblue')
    plt.plot(val_acc, label='Validation Accuracy', color='darkorange')
    plt.axvline(fine_tune_start - 1, linestyle='--', color='red', label='Start Fine-Tuning')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='royalblue')
    plt.plot(val_loss, label='Validation Loss', color='darkorange')
    plt.axvline(fine_tune_start - 1, linestyle='--', color='red', label='Start Fine-Tuning')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    history_image_path = os.path.join(output_dir, 'training_history_v2.png')
    plt.savefig(history_image_path, dpi=300, bbox_inches='tight')
    print(f"\nกราฟผลการฝึกสอนถูกบันทึกเป็นไฟล์ '{history_image_path}'")

# บันทึกผลลัพธ์
def log_run_summary_v2(run_dir, history1, history2, summary_file):
    """บันทึกข้อมูลสรุปของการฝึกสอน"""
    all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    best_val_acc = max(all_val_acc)
    best_epoch = all_val_acc.index(best_val_acc)
    corresponding_val_loss = all_val_loss[best_epoch]
    
    summary_data = {
        'run_id': os.path.basename(run_dir),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': 'EfficientNetB0_v2',
        'best_val_accuracy': f"{best_val_acc:.4f}",
        'val_loss_at_best_acc': f"{corresponding_val_loss:.4f}",
        'total_epochs': len(all_val_acc),
        'fine_tune_start': len(history1.history['val_accuracy'])
    }
    
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)
    
    print(f"\nข้อมูลสรุปถูกบันทึกไปยัง '{summary_file}'")
    print(f"ความแม่นยำสูงสุด: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

print("\n--- สรุปผลการฝึกสอน ---")
log_run_summary_v2(RUN_OUTPUT_DIR, history1, history2, SUMMARY_FILE)
plot_history_v2(history1, history2, RUN_OUTPUT_DIR)

print(f"\nโมเดลที่ดีที่สุดถูกบันทึกในชื่อ '{best_model_path}'")
print("การฝึกสอนเสร็จสิ้น!")