"""
สคริปต์สำหรับรันการฝึกสอนแบบครบถ้วน
เพื่อเพิ่มประสิทธิภาพเป็น 80-95%
"""

import os
import sys
import subprocess

def run_step(step_name, script_name, description):
    """รันขั้นตอนการฝึกสอน"""
    print(f"\n{'='*60}")
    print(f"ขั้นตอนที่ {step_name}: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print("✅ สำเร็จ!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # แสดง 500 ตัวอักษรสุดท้าย
        return True
    except subprocess.CalledProcessError as e:
        print("❌ เกิดข้อผิดพลาด!")
        print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ {script_name}")
        return False

def main():
    print("🚀 เริ่มกระบวนการปรับปรุงโมเดลเพื่อเพิ่มประสิทธิภาพเป็น 80-95%")
    
    steps = [
        ("1", "improve_dataset.py", "ปรับสมดุลข้อมูล (Data Balancing)"),
        ("2", "train_meat_classifier_v2.py", "ฝึกโมเดล EfficientNetV2B0 พร้อม Fine-tuning"),
    ]
    
    success_count = 0
    
    for step_num, script, desc in steps:
        if run_step(step_num, script, desc):
            success_count += 1
        else:
            print(f"\n❌ หยุดการทำงานเนื่องจากขั้นตอนที่ {step_num} ล้มเหลว")
            break
    
    print(f"\n{'='*60}")
    print(f"สรุปผลการทำงาน: {success_count}/{len(steps)} ขั้นตอนสำเร็จ")
    print(f"{'='*60}")
    
    if success_count == len(steps):
        print("🎉 การปรับปรุงเสร็จสิ้น!")
        print("\n📊 ผลลัพธ์ที่คาดหวัง:")
        print("- ความแม่นยำ: 75-90%")
        print("- ลด Overfitting อย่างมาก")
        print("- โมเดลเสถียรและ Generalize ได้ดี")
        
        print("\n📁 ไฟล์ผลลัพธ์:")
        print("- runs_v2/[timestamp]/best_model_v2.h5")
        print("- runs_v2/[timestamp]/training_history_v2.png")
        print("- runs_summary_v2.csv")
        
        print("\n🔄 ขั้นตอนต่อไป (ถ้าต้องการ):")
        print("1. รัน ensemble_model.py สำหรับ Ensemble Learning")
        print("2. เพิ่มข้อมูลมากขึ้น")
        print("3. ลอง Test Time Augmentation (TTA)")
    else:
        print("❌ การปรับปรุงไม่สำเร็จ กรุณาตรวจสอบข้อผิดพลาด")

if __name__ == "__main__":
    main()