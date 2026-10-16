import os
import sys
import time
from pynput import mouse

# 1. บังคับโหลดไฟล์ DLL/PYD จากโฟลเดอร์ปัจจุบัน
current_dir = os.path.dirname(os.path.abspath(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)
sys.path.append(current_dir)

import tobii_research as tr

# กำหนดความละเอียดหน้าจอแบบคงที่ (หรือใช้ไลบรารีอื่นช่วยดึงสเกลจอ)
# ตัวอย่าง: จอ Full HD
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ตัวแปรระดับ Global สำหรับเก็บพิกัดเมาส์ล่าสุด
current_mouse_x = 0
current_mouse_y = 0

# --- 2. ฟังก์ชันดักจับพิกัดเมาส์ด้วย pynput ---
def on_move(x, y):
    global current_mouse_x, current_mouse_y
    current_mouse_x = int(x)
    current_mouse_y = int(y)

# เริ่มเปิด Listener ของเมาส์ให้ทำงานเบื้องหลัง
mouse_listener = mouse.Listener(on_move=on_move)
mouse_listener.start()

# --- 3. ฟังก์ชันดักจับพิกัดสายตาด้วย Tobii ---
def gaze_data_callback(gaze_data):
    # ดึงพิกัดสายตาซ้ายและขวา (ค่าสัดส่วน 0.0 ถึง 1.0)
    left_eye = gaze_data['left_gaze_point_on_display_area']
    right_eye = gaze_data['right_gaze_point_on_display_area']
    
    # หาค่าเฉลี่ยสัดส่วนทิศทางการมอง
    gaze_ratio_x = (left_eye[0] + right_eye[0]) / 2
    gaze_ratio_y = (left_eye[1] + right_eye[1]) / 2
    
    # แปลงพิกัดสายตาให้กลายเป็นพิกเซลตามขนาดหน้าจอ
    gaze_pixel_x = int(gaze_ratio_x * SCREEN_WIDTH)
    gaze_pixel_y = int(gaze_ratio_y * SCREEN_HEIGHT)
    
    # แสดงผลเปรียบเทียบกับตัวแปรเมาส์ที่ถูก pynput อัปเดตตลอดเวลา
    print(f"👀 สายตา (Px): X={gaze_pixel_x:4d}, Y={gaze_pixel_y:4d} | 🖱️ เมาส์ (Px): X={current_mouse_x:4d}, Y={current_mouse_y:4d}", end="\r")

# --- 4. เริ่มระบบการทำงานของ Tobii ---
found_eyetrackers = tr.find_all_eyetrackers()

if not found_eyetrackers:
    print("ไม่พบเครื่อง Tobii Eye Tracker กรุณาตรวจสอบฮาร์ดแวร์อีกครั้ง")
    mouse_listener.stop()
    exit()

eyetracker = found_eyetrackers[0]
print(f"เชื่อมต่อสำเร็จ: {eyetracker.device_name}")

# เริ่มลงทะเบียนดักจับข้อมูลสายตา
eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

try:
    print("ระบบกำลังทำงาน... (กด Ctrl + C ใน Terminal เพื่อหยุดการทำงาน)")
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    # เคลียร์ระบบเมื่อปิดโปรแกรม
    eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    mouse_listener.stop()
    print("\nหยุดบันทึกพิกัดเรียบร้อยแล้ว")