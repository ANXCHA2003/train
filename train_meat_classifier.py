import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import json
import os
from datetime import datetime
import csv

# --- 1. กำหนดค่าพื้นฐาน ---

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

TEST_SPLIT = 0.2 # แบ่งข้อมูลสำหรับ Validation 20%

RANDOM_STATE = 42 # ทำให้การแบ่งข้อมูลเหมือนเดิมทุกครั้ง



DATA_DIR = 'meat database'

INITIAL_EPOCHS = 10

FINE_TUNE_EPOCHS = 20

SUMMARY_FILE = 'runs_summary.csv'



# สร้างโฟลเดอร์สำหรับจัดเก็บผลลัพธ์ของการรันครั้งนี้

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_OUTPUT_DIR = os.path.join('runs', timestamp)

os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

print(f"ผลลัพธ์ของการฝึกสอนครั้งนี้จะถูกเก็บไว้ที่: {RUN_OUTPUT_DIR}")





# --- 2. เตรียมข้อมูลรูปภาพ (แก้ไขใหม่) ---

print("\n--- เริ่มกระบวนการเตรียมข้อมูล ---")



# 2.1 ค้นหาไฟล์และสร้าง Labels

print("ค้นหาชื่อคลาสและไฟล์รูปภาพทั้งหมด...")



# ค้นหาชื่อคลาสจากโฟลเดอร์ย่อยใน 'train' เพื่อความถูกต้อง

class_names = sorted([os.path.basename(d) for d in glob.glob(os.path.join(DATA_DIR, 'train', '*')) if os.path.isdir(d)])

if not class_names:

    raise ValueError(f"ไม่พบคลาสในโฟลเดอร์ {os.path.join(DATA_DIR, 'train')}")

num_classes = len(class_names)

class_to_int = {name: i for i, name in enumerate(class_names)}

int_to_class = {i: name for i, name in enumerate(class_names)}



print(f"พบคลาสทั้งหมด {num_classes} คลาส: {class_names}")



# บันทึก Class Labels

class_labels_path = os.path.join(RUN_OUTPUT_DIR, 'class_labels.json')

with open(class_labels_path, 'w') as f:

    json.dump(int_to_class, f)

print(f"บันทึก Class Labels เรียบร้อยแล้ว: {int_to_class}")



# ค้นหาไฟล์รูปภาพทั้งหมดในทุกโฟลเดอร์ย่อยแบบ Recursive

image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True)

if not image_paths:

    raise ValueError(f"ไม่พบไฟล์ .jpg ในโฟลเดอร์ {DATA_DIR}")



# สร้าง list ของ labels ให้สอดคล้องกับ image_paths ที่ถูกต้องเท่านั้น

labels = []

valid_image_paths = []

for path in image_paths:

    label_name = os.path.basename(os.path.dirname(path))

    if label_name in class_to_int:

        labels.append(class_to_int[label_name])

        valid_image_paths.append(path)



print(f"พบรูปภาพที่ตรงกับคลาสทั้งหมด {len(valid_image_paths)} รูป")



# 2.2 แบ่งข้อมูล Train / Validation แบบ Fix

print(f"แบ่งข้อมูล Train/Validation ด้วยอัตราส่วน {1-TEST_SPLIT:.0%}:{TEST_SPLIT:.0%} และ random_state={RANDOM_STATE}")

train_paths, val_paths, train_labels, val_labels = train_test_split(

    valid_image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE, shuffle=True, stratify=labels

)

print(f"จำนวนข้อมูล Train: {len(train_paths)}, Validation: {len(val_paths)}")



# 2.3 โหลดและแปลงรูปภาพเป็น Numpy Arrays

def load_and_preprocess_images(paths, labels, image_size, num_classes):

    images_np = np.zeros((len(paths), *image_size, 3), dtype=np.float32)

    labels_np = np.array(labels, dtype=np.int32)

    

    for i, path in enumerate(paths):

        img = load_img(path, target_size=image_size)

        img_array = img_to_array(img)

        images_np[i] = img_array

        

    # Rescale

    images_np /= 255.0

    

    # Convert labels to one-hot encoding

    labels_np = to_categorical(labels_np, num_classes=num_classes)

    

    return images_np, labels_np



print("กำลังโหลดและแปลงรูปภาพสำหรับ Training set...")

x_train, y_train = load_and_preprocess_images(train_paths, train_labels, IMAGE_SIZE, num_classes)

print("กำลังโหลดและแปลงรูปภาพสำหรับ Validation set...")

x_val, y_val = load_and_preprocess_images(val_paths, val_labels, IMAGE_SIZE, num_classes)



print("--- การเตรียมข้อมูลเสร็จสิ้น ---")



# 2.4 สร้าง Data Generator สำหรับ Augmentation (เฉพาะข้อมูล Train)

train_datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)



train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)



# Validation data ไม่ต้องมี Generator เพราะเราจะใช้ข้อมูลที่โหลดมาโดยตรง

validation_data = (x_val, y_val)




# --- 3. สร้างโมเดล (Transfer Learning) ---
# โหลด MobileNetV2 ที่เรียนรู้จากชุดข้อมูล ImageNet มาแล้ว
# include_top=False คือเราไม่เอาชั้นบนสุด (ชั้นที่ใช้จำแนก 1000 สิ่ง) ออก
# เราจะสร้างชั้นบนสุดขึ้นมาใหม่เพื่อจำแนกแค่ "ส้ม" กับ "ไม่ใช่ส้ม"
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# "แช่แข็ง" โมเดลพื้นฐาน ไม่ให้มันเรียนรู้ใหม่ทั้งหมด
base_model.trainable = False

# สร้างชั้นบนสุดขึ้นมาใหม่
x = base_model.output
x = GlobalAveragePooling2D()(x) # ลดขนาดข้อมูล
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # เพิ่ม Dropout เพื่อช่วยลด Overfitting
# ชั้นสุดท้ายมี num_classes neuron (จำนวนคลาส) และใช้ softmax activation
# เพื่อให้ผลลัพธ์เป็นความน่าจะเป็นของแต่ละคลาส
predictions = Dense(num_classes, activation='softmax')(x)

# รวมโมเดลเก่ากับชั้นใหม่เข้าด้วยกัน
model = Model(inputs=base_model.input, outputs=predictions)

# --- บันทึก Class Indices ---
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}
class_labels_path = os.path.join(RUN_OUTPUT_DIR, 'class_labels.json')
with open(class_labels_path, 'w') as f:
    json.dump(labels, f)
print(f"บันทึก Class Labels เรียบร้อยแล้ว: {labels}")

# --- 4. คอมไพล์และฝึกสอนโมเดล ---
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# กำหนดเส้นทางสำหรับบันทึกโมเดลที่ดีที่สุด
best_model_path = os.path.join(RUN_OUTPUT_DIR, 'best_model.h5')

# สร้าง Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)

# เริ่มการฝึกสอน
history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_data,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# --- 5. Fine-Tuning ---
print("\n--- เริ่มกระบวนการ Fine-Tuning ---")

# ปลดล็อกโมเดลพื้นฐาน
base_model.trainable = True

# ปลดล็อกเฉพาะบางส่วนท้ายๆ ของโมเดล (เช่น 50 layers สุดท้าย)
# การปลดล็อกทั้งหมดอาจทำให้โมเดลเสียหายได้
fine_tune_at = 150
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# คอมไพล์โมเดลอีกครั้งด้วย Learning Rate ที่ต่ำมากๆ
# เพื่อป้องกันไม่ให้ความรู้เดิมของโมเดลถูกทำลาย
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# โหลดโมเดลที่ดีที่สุดที่บันทึกไว้กลับมาเพื่อเริ่ม Fine-tuning
print("\nโหลดโมเดลที่ดีที่สุดกลับมาเพื่อเริ่ม Fine-tuning...")
model.load_weights(best_model_path)

print(f"ทำการ Fine-tune โมเดล โดยปลดล็อกตั้งแต่ Layer ที่ {fine_tune_at} เป็นต้นไป")

# ฝึกสอนต่อ (Fine-tune)
total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1, # เริ่มต้นฝึกต่อจากรอบที่แล้ว
    validation_data=validation_data,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler] # ใช้ Callbacks เดิม
)

# --- 6. ประเมินผลและแสดงกราฟ ---
def plot_history(history, history_fine, output_dir):
    """
    พล็อตกราฟแสดง Accuracy และ Loss จากการฝึกสอนโมเดล และบันทึกเป็นไฟล์
    """
    # ใช้สไตล์เพื่อความสวยงาม
    plt.style.use('seaborn-v0_8-whitegrid')

    # รวม history จากทั้ง 2 เฟส
    acc = history.history.get('accuracy', []) + history_fine.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', []) + history_fine.history.get('val_accuracy', [])
    loss = history.history.get('loss', []) + history_fine.history.get('loss', [])
    val_loss = history.history.get('val_loss', []) + history_fine.history.get('val_loss', [])

    # จุดที่เริ่มทำ Fine-tuning (แม่นยำกว่าการใช้ค่าคงที่)
    fine_tune_start_epoch = len(history.history.get('accuracy', []))

    # พล็อตกราฟ Accuracy
    plt.figure(figsize=(16, 6))
    plt.suptitle('Training and Validation History', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='royalblue')
    plt.plot(val_acc, label='Validation Accuracy', color='darkorange')
    plt.axvline(fine_tune_start_epoch - 1, linestyle='--', color='gray', label='Start Fine-Tuning')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # พล็อตกราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='royalblue')
    plt.plot(val_loss, label='Validation Loss', color='darkorange')
    plt.axvline(fine_tune_start_epoch - 1, linestyle='--', color='gray', label='Start Fine-Tuning')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # บันทึกกราฟเป็นไฟล์
    history_image_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_image_path)
    print(f"\nกราฟผลการฝึกสอนถูกบันทึกเป็นไฟล์ '{history_image_path}'")
    # plt.show() # แสดงกราฟแบบ interactive (ถูกปิดใช้งาน)

# --- 7. บันทึกโมเดลที่ฝึกเสร็จแล้ว ---
def log_run_summary(run_dir, history, history_fine, summary_file):
    """บันทึกข้อมูลสรุปของการฝึกสอนลงในไฟล์ CSV"""
    # หาค่าที่ดีที่สุดจาก Callbacks
    best_val_acc = max(history.history.get('val_accuracy', [0]) + history_fine.history.get('val_accuracy', [0]))
    
    # หา val_loss ที่สอดคล้องกับ best_val_acc
    all_val_acc = history.history.get('val_accuracy', []) + history_fine.history.get('val_accuracy', [])
    all_val_loss = history.history.get('val_loss', []) + history_fine.history.get('val_loss', [])
    best_epoch_index = all_val_acc.index(best_val_acc)
    corresponding_val_loss = all_val_loss[best_epoch_index]

    summary_data = {
        'run_id': os.path.basename(run_dir),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'best_val_accuracy': f"{best_val_acc:.4f}",
        'val_loss_at_best_acc': f"{corresponding_val_loss:.4f}"
    }

    file_exists = os.path.isfile(summary_file)
    with open(summary_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader() # เขียนหัวข้อคอลัมน์ถ้าไฟล์ยังไม่มี
        writer.writerow(summary_data)
    print(f"\nข้อมูลสรุปถูกบันทึกไปยัง '{summary_file}'")

print("กระบวนการฝึกสอนเสร็จสิ้น")
print(f"โมเดลที่ดีที่สุดถูกบันทึกในชื่อ '{best_model_path}'")
log_run_summary(RUN_OUTPUT_DIR, history, history_fine, SUMMARY_FILE)
plot_history(history, history_fine, RUN_OUTPUT_DIR)
