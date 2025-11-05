import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import csv

# --- 1. กำหนดค่าพื้นฐาน ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16 # ลด Batch Size อาจช่วยให้โมเดลเรียนรู้ได้ดีขึ้นบนข้อมูลที่ไม่ใหญ่มาก
TRAIN_DIR = 'meat database/train'
VALID_DIR = 'meat database/validation'
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
SUMMARY_FILE = 'runs_summary.csv'

# สร้างโฟลเดอร์สำหรับจัดเก็บผลลัพธ์ของการรันครั้งนี้
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR = os.path.join('runs', timestamp)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
print(f"ผลลัพธ์ของการฝึกสอนครั้งนี้จะถูกเก็บไว้ที่: {RUN_OUTPUT_DIR}")


# --- 2. เตรียมข้อมูลรูปภาพ ---
# สร้างเครื่องมือเพื่ออ่านและแปลงรูปภาพจากโฟลเดอร์
# rescale=1./255 คือการปรับค่าสีของภาพจาก 0-255 ให้อยู่ในช่วง 0-1 (Normalization)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25, # เพิ่มความหลากหลายของข้อมูลโดยการหมุนภาพ
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# ดึงข้อมูลจากโฟลเดอร์มาเตรียมไว้
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # 'binary' เพราะมีแค่ 2 คลาส (orange, not_orange)
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 3. สร้างโมเดล (Transfer Learning) ---
# โหลด MobileNetV2 ที่เรียนรู้จากชุดข้อมูล ImageNet มาแล้ว
# include_top=False คือเราไม่เอาชั้นบนสุด (ชั้นที่ใช้จำแนก 1000 สิ่ง) ออก
# เราจะสร้างชั้นบนสุดขึ้นมาใหม่เพื่อจำแนกแค่ "ส้ม" กับ "ไม่ใช่ส้ม"
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# "แช่แข็ง" โมเดลพื้นฐาน ไม่ให้มันเรียนรู้ใหม่ทั้งหมด
base_model.trainable = False

# สร้างชั้นบนสุดขึ้นมาใหม่
x = base_model.output
x = GlobalAveragePooling2D()(x) # ลดขนาดข้อมูล
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # เพิ่ม Dropout เพื่อช่วยลด Overfitting
# ชั้นสุดท้ายมี 1 neuron และใช้ sigmoid activation เพื่อให้ผลลัพธ์เป็นค่าระหว่าง 0-1
# (ค่าใกล้ 1 คือส้ม, ค่าใกล้ 0 คือไม่ใช่ส้ม)
predictions = Dense(1, activation='sigmoid')(x)

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
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# กำหนดเส้นทางสำหรับบันทึกโมเดลที่ดีที่สุด
best_model_path = os.path.join(RUN_OUTPUT_DIR, 'best_model.h5')

# สร้าง Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

# เริ่มการฝึกสอน
history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# --- 5. Fine-Tuning ---
print("\n--- เริ่มกระบวนการ Fine-Tuning ---")

# ปลดล็อกโมเดลพื้นฐาน
base_model.trainable = True

# ปลดล็อกเฉพาะบางส่วนท้ายๆ ของโมเดล (เช่น 50 layers สุดท้าย)
# การปลดล็อกทั้งหมดอาจทำให้โมเดลเสียหายได้
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# คอมไพล์โมเดลอีกครั้งด้วย Learning Rate ที่ต่ำมากๆ
# เพื่อป้องกันไม่ให้ความรู้เดิมของโมเดลถูกทำลาย
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
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
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint] # ใช้ Callbacks เดิม
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
    plt.show()

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
plot_history(history, history_fine, RUN_OUTPUT_DIR)
log_run_summary(RUN_OUTPUT_DIR, history, history_fine, SUMMARY_FILE)
