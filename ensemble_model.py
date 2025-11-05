import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50V2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import numpy as np
import json

def create_ensemble_model(num_classes, image_size=(240, 240, 3)):
    """สร้าง Ensemble Model จาก 3 โมเดลที่แตกต่างกัน"""
    
    # Model 1: EfficientNetV2B0
    base1 = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=image_size)
    base1.trainable = False
    
    x1 = base1.output
    x1 = GlobalAveragePooling2D()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(0.3)(x1)
    output1 = Dense(num_classes, activation='softmax', name='efficientnet_output')(x1)
    
    model1 = Model(inputs=base1.input, outputs=output1)
    
    # Model 2: ResNet50V2
    base2 = ResNet50V2(weights='imagenet', include_top=False, input_shape=image_size)
    base2.trainable = False
    
    x2 = base2.output
    x2 = GlobalAveragePooling2D()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.3)(x2)
    output2 = Dense(num_classes, activation='softmax', name='resnet_output')(x2)
    
    model2 = Model(inputs=base2.input, outputs=output2)
    
    # Model 3: DenseNet121
    base3 = DenseNet121(weights='imagenet', include_top=False, input_shape=image_size)
    base3.trainable = False
    
    x3 = base3.output
    x3 = GlobalAveragePooling2D()(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(0.3)(x3)
    output3 = Dense(num_classes, activation='softmax', name='densenet_output')(x3)
    
    model3 = Model(inputs=base3.input, outputs=output3)
    
    # Ensemble: Average predictions
    input_layer = tf.keras.Input(shape=image_size)
    
    pred1 = model1(input_layer)
    pred2 = model2(input_layer)
    pred3 = model3(input_layer)
    
    # Average the predictions
    ensemble_output = Average()([pred1, pred2, pred3])
    
    ensemble_model = Model(inputs=input_layer, outputs=ensemble_output)
    
    return ensemble_model, [model1, model2, model3]

def train_ensemble_step_by_step(x_train, y_train, x_val, y_val, num_classes, 
                               class_weight_dict, output_dir):
    """ฝึก Ensemble Model ทีละโมเดล"""
    
    print("=== การฝึก Ensemble Model ===")
    
    # สร้าง ensemble
    ensemble_model, individual_models = create_ensemble_model(num_classes)
    
    trained_models = []
    
    # ฝึกทีละโมเดล
    model_names = ['EfficientNetV2B0', 'ResNet50V2', 'DenseNet121']
    
    for i, (model, name) in enumerate(zip(individual_models, model_names)):
        print(f"\n--- ฝึกโมเดล {i+1}: {name} ---")
        
        model.compile(
            optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=15, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3, 
                patience=8, 
                min_lr=1e-7
            )
        ]
        
        # ฝึกโมเดล
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # บันทึกโมเดล
        model_path = os.path.join(output_dir, f'{name.lower()}_model.h5')
        model.save(model_path)
        
        trained_models.append(model)
        
        # ประเมินผล
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        print(f"{name} - Validation Accuracy: {val_acc:.4f}")
    
    # ประเมิน Ensemble
    print("\n--- ประเมิน Ensemble Model ---")
    
    # Predict จากแต่ละโมเดล
    pred1 = trained_models[0].predict(x_val, verbose=0)
    pred2 = trained_models[1].predict(x_val, verbose=0)
    pred3 = trained_models[2].predict(x_val, verbose=0)
    
    # Average predictions
    ensemble_pred = (pred1 + pred2 + pred3) / 3
    
    # คำนวณ accuracy
    ensemble_acc = np.mean(np.argmax(ensemble_pred, axis=1) == np.argmax(y_val, axis=1))
    
    print(f"Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    
    # บันทึกผลลัพธ์
    results = {
        'individual_accuracies': {},
        'ensemble_accuracy': float(ensemble_acc),
        'model_paths': {}
    }
    
    for i, name in enumerate(model_names):
        val_loss, val_acc = trained_models[i].evaluate(x_val, y_val, verbose=0)
        results['individual_accuracies'][name] = float(val_acc)
        results['model_paths'][name] = f'{name.lower()}_model.h5'
    
    # บันทึกผลลัพธ์
    results_path = os.path.join(output_dir, 'ensemble_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nผลลัพธ์ Ensemble ถูกบันทึกใน: {results_path}")
    
    return trained_models, ensemble_acc

if __name__ == "__main__":
    print("สคริปต์นี้ใช้สำหรับสร้าง Ensemble Model")
    print("กรุณาเรียกใช้จากสคริปต์หลัก")