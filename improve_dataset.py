import os
import shutil
import random
from collections import Counter
import glob

def count_images_per_class(data_dir):
    """นับจำนวนรูปภาพในแต่ละคลาส"""
    class_counts = {}
    for class_name in os.listdir(os.path.join(data_dir, 'train')):
        class_path = os.path.join(data_dir, 'train', class_name)
        if os.path.isdir(class_path):
            image_files = glob.glob(os.path.join(class_path, '*.jpg')) + glob.glob(os.path.join(class_path, '*.png'))
            class_counts[class_name] = len(image_files)
    return class_counts

def balance_dataset(data_dir, target_samples=200):
    """ปรับสมดุลข้อมูลให้แต่ละคลาสมีจำนวนใกล้เคียงกัน"""
    print("=== การปรับสมดุลข้อมูล ===")
    
    # นับจำนวนรูปภาพปัจจุบัน
    class_counts = count_images_per_class(data_dir)
    print("จำนวนรูปภาพปัจจุบัน:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} รูป")
    
    # สร้างโฟลเดอร์ balanced
    balanced_dir = os.path.join(data_dir, 'balanced_train')
    os.makedirs(balanced_dir, exist_ok=True)
    
    for class_name, count in class_counts.items():
        class_path = os.path.join(data_dir, 'train', class_name)
        balanced_class_path = os.path.join(balanced_dir, class_name)
        os.makedirs(balanced_class_path, exist_ok=True)
        
        # ค้นหาไฟล์รูปภาพ
        image_files = glob.glob(os.path.join(class_path, '*.jpg')) + glob.glob(os.path.join(class_path, '*.png'))
        
        if count > target_samples:
            # ถ้ามีมากเกินไป ให้สุ่มเลือก
            selected_files = random.sample(image_files, target_samples)
            print(f"  {class_name}: ลดจาก {count} เป็น {target_samples} รูป")
        else:
            # ถ้ามีน้อย ให้ใช้ทั้งหมด
            selected_files = image_files
            print(f"  {class_name}: ใช้ทั้งหมด {count} รูป")
        
        # คัดลอกไฟล์ที่เลือก
        for i, src_file in enumerate(selected_files):
            dst_file = os.path.join(balanced_class_path, f"{class_name}_{i:03d}.jpg")
            shutil.copy2(src_file, dst_file)
    
    print(f"\nข้อมูลที่ปรับสมดุลแล้วถูกบันทึกใน: {balanced_dir}")
    return balanced_dir

if __name__ == "__main__":
    # ปรับสมดุลข้อมูล
    balanced_dir = balance_dataset('meat database', target_samples=150)
    
    # นับจำนวนใหม่
    print("\n=== ผลลัพธ์หลังปรับสมดุล ===")
    new_counts = count_images_per_class('meat database')
    for class_name in new_counts:
        balanced_path = os.path.join(balanced_dir, class_name)
        if os.path.exists(balanced_path):
            balanced_count = len(os.listdir(balanced_path))
            print(f"  {class_name}: {balanced_count} รูป")