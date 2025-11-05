import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# --- กำหนดค่าพื้นฐาน ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# ใช้ข้อมูลที่ปรับสมดุลแล้ว
DATA_DIR = 'meat database/balanced_train'

# สร้างโฟลเดอร์สำหรับจัดเก็บผลลัพธ์
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR = os.path.join('runs_simple', timestamp)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
print(f"ผลลัพธ์จะถูกเก็บไว้ที่: {RUN_OUTPUT_DIR}")

# --- เตรียมข้อมูลด้วย ImageDataGenerator ---
print("\n--- เตรียมข้อมูลด้วย ImageDataGenerator ---")

# Data Augmentation แบบเรียบง่าย
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # แบ่งข้อมูล 80% train, 20% validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# สร้าง data generators
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"พบคลาสทั้งหมด {num_classes} คลาส: {list(train_generator.class_indices.keys())}")
print(f"จำนวนข้อมูล Train: {train_generator.samples}, Validation: {validation_generator.samples}")

# --- สร้างโมเดลแบบเรียบง่าย ---
print("\n--- สร้างโมเดล ---")

# ใช้ MobileNetV2 ที่เรียบง่ายกว่า
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# แช่แข็งโมเดลพื้นฐาน
base_model.trainable = False

# สร้างชั้นบนสุดที่เรียบง่าย
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- คอมไพล์โมเดล ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("โครงสร้างโมเดล:")
model.summary()

# --- Callbacks ---
best_model_path = os.path.join(RUN_OUTPUT_DIR, 'best_simple_model.h5')

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
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
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    )
]

# --- ฝึกสอนโมเดล ---
print("\n--- เริ่มการฝึกสอน ---")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# --- ประเมินผล ---
print("\n--- ประเมินผล ---")

# ประเมินบน validation set
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Validation Loss: {val_loss:.4f}")

# --- สร้างกราฟ ---
def plot_training_history(history, output_dir):
    """พล็อตกราฟแสดงผลการฝึกสอน"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # บันทึกกราฟ
    history_image_path = os.path.join(output_dir, 'training_history_simple.png')
    plt.savefig(history_image_path, dpi=300, bbox_inches='tight')
    print(f"\nกราฟผลการฝึกสอนถูกบันทึกเป็นไฟล์ '{history_image_path}'")
    plt.show()

plot_training_history(history, RUN_OUTPUT_DIR)

# --- บันทึกผลลัพธ์ ---
import json

results = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'model': 'MobileNetV2_simple',
    'val_accuracy': float(val_accuracy),
    'val_loss': float(val_loss),
    'epochs_trained': len(history.history['accuracy']),
    'class_indices': train_generator.class_indices
}

results_path = os.path.join(RUN_OUTPUT_DIR, 'results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nผลลัพธ์ถูกบันทึกใน: {results_path}")
print(f"โมเดลที่ดีที่สุดถูกบันทึกใน: {best_model_path}")
print("\n=== การฝึกสอนเสร็จสิ้น ===")
print(f"ความแม่นยำสุดท้าย: {val_accuracy*100:.2f}%")