import os
import sys
import time
import csv
import math
import random
from pynput import mouse

# 1. ตั้งค่าระบบเพื่อโหลดไฟล์ DLL/PYD ของ Tobii จากไดเรกทอรีปัจจุบัน
current_dir = os.path.dirname(os.path.abspath(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)
sys.path.append(current_dir)

try:
    import tobii_research as tr
    TOBII_AVAILABLE = True
    TOBII_IMPORT_ERROR = None
except Exception as exc:
    tr = None
    TOBII_AVAILABLE = False
    TOBII_IMPORT_ERROR = exc

# 2. ตั้งค่าขนาดหน้าจอคอมพิวเตอร์ (ปรับเปลี่ยนให้ตรงกับหน้าจอที่ใช้ทดสอบจริง)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

args = sys.argv[1:]
max_samples = None
subject_name = None
for index, arg in enumerate(args):
    if arg == '--max-samples' and index + 1 < len(args):
        max_samples = int(args[index + 1])
    elif arg == '--subject-name' and index + 1 < len(args):
        subject_name = args[index + 1]

print("==================================================")
print("    โปรแกรมบันทึกข้อมูล Eye & Mouse Tracking (ใช้งานจริง)   ")
print("==================================================")

# 3. ระบบถามชื่อผู้ทดสอบเพื่อนำไปตั้งชื่อไฟล์แยกของใครของมัน
if subject_name is None:
    if sys.stdin.isatty():
        try:
            subject_name = input("กรุณากรอกชื่อหรือรหัสผู้ทดสอบ (เช่น User01): ").strip()
        except EOFError:
            subject_name = ""
        except KeyboardInterrupt:
            subject_name = ""
    else:
        try:
            subject_name = sys.stdin.readline().strip()
        except Exception:
            subject_name = ""

if not subject_name:
    subject_name = "unknown_user"

safe_subject_name = ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in subject_name)
filename = f'dataset_{safe_subject_name}.csv'
print(f"กำลังเก็บข้อมูลสำหรับผู้ใช้: {subject_name}")
print(f"ไฟล์ที่จะบันทึก: {filename}")

# เตรียมไฟล์ CSV พร้อมคอลัมน์ทั้งหมด 11 Features
csv_file = open(filename, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamp',          # Eye-Tracking 1
    'gaze_x',             # Eye-Tracking 2
    'gaze_y',             # Eye-Tracking 3
    'pupil_left',         # Eye-Tracking 4
    'pupil_right',        # Eye-Tracking 5
    'fixation_duration',  # Eye-Tracking 6
    'saccade_amplitude',  # Eye-Tracking 7
    'mouse_x',            # Mouse-Tracking 1
    'mouse_y',            # Mouse-Tracking 2
    'click_event',        # Mouse-Tracking 3
    'scroll_dis'          # Mouse-Tracking 4
])

# ตัวแปร Global สำหรับเก็บค่า Mouse-Tracking ปัจจุบัน
current_mouse_x = 0
current_mouse_y = 0
current_click_event = "None"
current_scroll_dis = 0

# ตัวแปร Global สำหรับคำนวณสถิติสายตา
last_gaze_x = SCREEN_WIDTH // 2
last_gaze_y = SCREEN_HEIGHT // 2
fixation_start_time = None
current_fixation_duration = 0
start_time = None
sample_counter = 0

# --- 4. ฟังก์ชันดักจับเมาส์เชิงลึกด้วย pynput ---
def on_move(x, y):
    global current_mouse_x, current_mouse_y
    current_mouse_x = int(x)
    current_mouse_y = int(y)

def on_click(x, y, button, pressed):
    global current_click_event
    if pressed:
        if button == mouse.Button.left:
            current_click_event = "Left_Click"
        elif button == mouse.Button.right:
            current_click_event = "Right_Click"
        else:
            current_click_event = f"{button.name}_Click"
    else:
        current_click_event = "None"

def on_scroll(x, y, dx, dy):
    global current_scroll_dis
    current_scroll_dis = int(dy)

# เริ่มเปิดใช้งาน Listener สำหรับเมาส์
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
mouse_listener.start()

# --- 5. ฟังก์ชันดักจับและคำนวณข้อมูลสายตาจริงจาก Tobii SDK ---
def gaze_data_callback(gaze_data):
    global last_gaze_x, last_gaze_y, fixation_start_time, current_fixation_duration, current_scroll_dis, start_time, sample_counter
    
    if start_time is None:
        start_time = time.time()
    timestamp = round(time.time() - start_time, 3)
    
    # ดึงค่าพิกัดสายตาเฉลี่ยซ้าย-ขวา
    left_eye_gaze = gaze_data['left_gaze_point_on_display_area']
    right_eye_gaze = gaze_data['right_gaze_point_on_display_area']
    
    gaze_ratio_x = (left_eye_gaze[0] + right_eye_gaze[0]) / 2
    gaze_ratio_y = (left_eye_gaze[1] + right_eye_gaze[1]) / 2
    
    # แปลงสัดส่วนเป็นพิกเซลหน้าจอ
    gaze_x = int(gaze_ratio_x * SCREEN_WIDTH)
    gaze_y = int(gaze_ratio_y * SCREEN_HEIGHT)
    
    # ดึงขนาดรูม่านตาซ้ายและขวา (หน่วยเป็นมิลลิเมตรจากเครื่องจริง)
    pupil_left = round(gaze_data['left_pupil_diameter'], 2)
    pupil_right = round(gaze_data['right_pupil_diameter'], 2)
    
    #คำนวณ Saccade Amplitude (ระยะการกวาดสายตาจากจุดก่อนหน้าเป็นพิกเซล)
    saccade_amplitude = round(math.sqrt((gaze_x - last_gaze_x)**2 + (gaze_y - last_gaze_y)**2), 2)
    
    # คำนวณ Fixation Duration เบื้องต้น (ถ้าสายตาขยับน้อยกว่า 30 พิกเซล ถือว่ากำลังจ้องจุดเดิม)
    if saccade_amplitude < 30:
        if fixation_start_time is None:
            fixation_start_time = timestamp
        current_fixation_duration = int((timestamp - fixation_start_time) * 1000) # แปลงเป็นมิลลิวินาที
    else:
        fixation_start_time = None
        current_fixation_duration = 0

    # บันทึกข้อมูลจริงทั้ง 11 Features ลงไฟล์ CSV
    csv_writer.writerow([
        timestamp, gaze_x, gaze_y, pupil_left, pupil_right, current_fixation_duration, saccade_amplitude,
        current_mouse_x, current_mouse_y, current_click_event, current_scroll_dis
    ])
    
    # มอนิเตอร์ค่าผ่าน Terminal
    print(
        f"บันทึกไฟล์: {filename} | Gaze:({gaze_x},{gaze_y}) | "
        f"Mouse:({current_mouse_x},{current_mouse_y}) Click:{current_click_event}", 
        end="\r"
    )
    
    # อัปเดตพิกัดอ้างอิงรอบถัดไป และรีเซ็ตค่าสกรอลล์
    last_gaze_x = gaze_x
    last_gaze_y = gaze_y
    current_scroll_dis = 0

# --- 6. เริ่มต้นเชื่อมต่อฮาร์ดแวร์ Tobii หรือเข้าสู่โหมดจำลอง ---
if TOBII_AVAILABLE:
    print("\nกำลังค้นหาเครื่อง Tobii Eye Tracker...")
    try:
        found_eyetrackers = tr.find_all_eyetrackers()
    except Exception as exc:
        print(f"ไม่สามารถเชื่อมต่อ Tobii ได้: {exc}")
        found_eyetrackers = []

    if not found_eyetrackers:
        print("ไม่พบเครื่อง Tobii Eye Tracker กรุณาตรวจสอบการเชื่อมต่อฮาร์ดแวร์")
        print("กำลังเข้าสู่โหมดจำลองเพราะ Tobii SDK ไม่พร้อม")
        TOBII_AVAILABLE = False

if not TOBII_AVAILABLE:
    if TOBII_IMPORT_ERROR is not None:
        print(f"Tobii SDK ไม่พร้อม: {TOBII_IMPORT_ERROR}")
    print("▶เริ่มทำการเก็บข้อมูลในโหมดจำลอง... (กด [Ctrl + C] เพื่อหยุดและเซฟไฟล์)")

    sample_count = 0
    try:
        while max_samples is None or sample_count < max_samples:
            if start_time is None:
                start_time = time.time()
            timestamp = round(time.time() - start_time, 3)

            gaze_x = int(max(0, min(SCREEN_WIDTH, last_gaze_x + random.randint(-80, 80))))
            gaze_y = int(max(0, min(SCREEN_HEIGHT, last_gaze_y + random.randint(-60, 60))))

            pupil_left = round(random.uniform(2.5, 3.8), 2)
            pupil_right = round(random.uniform(2.5, 3.8), 2)

            saccade_amplitude = round(math.sqrt((gaze_x - last_gaze_x) ** 2 + (gaze_y - last_gaze_y) ** 2), 2)

            if saccade_amplitude < 30:
                if fixation_start_time is None:
                    fixation_start_time = timestamp
                current_fixation_duration = int((timestamp - fixation_start_time) * 1000)
            else:
                fixation_start_time = None
                current_fixation_duration = 0

            csv_writer.writerow([
                timestamp, gaze_x, gaze_y, pupil_left, pupil_right, current_fixation_duration, saccade_amplitude,
                current_mouse_x, current_mouse_y, current_click_event, current_scroll_dis
            ])

            print(
                f"บันทึกไฟล์: {filename} | Gaze:({gaze_x},{gaze_y}) | "
                f"Mouse:({current_mouse_x},{current_mouse_y}) Click:{current_click_event}",
                end="\r"
            )

            last_gaze_x = gaze_x
            last_gaze_y = gaze_y
            current_scroll_dis = 0
            sample_count += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        mouse_listener.stop()
        csv_file.close()
        print(f"\n\n [สำเร็จ] บันทึกข้อมูลลงในไฟล์ '{filename}' เรียบร้อยแล้วครับ!")
else:
    eyetracker = found_eyetrackers[0]
    print(f"เชื่อมต่อสำเร็จ: {eyetracker.device_name}")
    print("▶เริ่มทำการเก็บข้อมูลจริง... (กด [Ctrl + C] เพื่อหยุดและเซฟไฟล์)")

    # ลงทะเบียนเริ่มต้นสตรีมข้อมูลจากเครื่องจริง
    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # 7. ปิดระบบเมื่อสิ้นสุดการทดลอง
        eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        mouse_listener.stop()
        csv_file.close()
        print(f"\n\n [สำเร็จ] บันทึกข้อมูลจริงลงในไฟล์ '{filename}' เรียบร้อยแล้วครับ!")

"""import os
import sys
import time
import csv
import random
import math
from pynput import mouse

# 1. ตั้งค่าระบบเพื่อความสมบูรณ์ของโครงสร้างพาธ
current_dir = os.path.dirname(os.path.abspath(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(current_dir)
sys.path.append(current_dir)

# 2. ตั้งค่าหน้าจอ
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

print("==================================================")
print("    โปรแกรมจำลองการเก็บข้อมูล Eye & Mouse Tracking    ")
print("==================================================")

# 3. ระบบถามชื่อผู้ทดสอบเพื่อนำไปตั้งชื่อไฟล์แยกของใครของมัน
subject_name = input("User01: ").strip()
if not subject_name:
    subject_name = "unknown_user"  # ป้องกันกรณีเผลอกด Enter ผ่านไปเฉย ๆ

# ตั้งชื่อไฟล์ตามชื่อผู้ทดสอบ
filename = f'dataset_{subject_name}.csv'

# เตรียมไฟล์ CSV พร้อมคอลัมน์ทั้งหมด 11 Features
csv_file = open(filename, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamp',          # Eye-Tracking 1
    'gaze_x',             # Eye-Tracking 2
    'gaze_y',             # Eye-Tracking 3
    'pupil_left',         # Eye-Tracking 4
    'pupil_right',        # Eye-Tracking 5
    'fixation_duration',  # Eye-Tracking 6
    'saccade_amplitude',  # Eye-Tracking 7
    'mouse_x',            # Mouse-Tracking 1
    'mouse_y',            # Mouse-Tracking 2
    'click_event',        # Mouse-Tracking 3
    'scroll_dis'          # Mouse-Tracking 4
])

# ตัวแปร Global สำหรับเก็บค่า Mouse-Tracking ปัจจุบัน
current_mouse_x = 0
current_mouse_y = 0
current_click_event = "None"
current_scroll_dis = 0

# --- 4. ฟังก์ชันดักจับเมาส์เชิงลึกด้วย pynput ---
def on_move(x, y):
    global current_mouse_x, current_mouse_y
    current_mouse_x = int(x)
    current_mouse_y = int(y)

def on_click(x, y, button, pressed):
    global current_click_event
    if pressed:
        if button == mouse.Button.left:
            current_click_event = "Left_Click"
        elif button == mouse.Button.right:
            current_click_event = "Right_Click"
        else:
            current_click_event = f"{button.name}_Click"
    else:
        current_click_event = "None"

def on_scroll(x, y, dx, dy):
    global current_scroll_dis
    current_scroll_dis = int(dy)

# เริ่มต้นเปิดใช้งาน Listener สำหรับเมาส์
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
mouse_listener.start()

# --- 5. ตัวแปรเสริมสำหรับจำลองคณิตศาสตร์สายตา ---
last_gaze_x = SCREEN_WIDTH // 2
last_gaze_y = SCREEN_HEIGHT // 2

print(f"\n⚙️ [โหมดจำลอง] เปิดระบบบันทึกไฟล์: {filename}")
print("▶️ เริ่มเก็บบันทึกข้อมูล... (ขยับเมาส์ คลิก หรือสกรอลล์เล่นได้เลย)")
print("🛑 เมื่อต้องการหยุดการทดสอบ ให้กด [Ctrl + C] ใน Terminal\n")

try:
    while True:
        timestamp = time.time()
        
        # --- จำลองพฤติกรรมสายตา (Eye-Tracking Features) ---
        gaze_x = int(max(0, min(SCREEN_WIDTH, last_gaze_x + random.randint(-80, 80))))
        gaze_y = int(max(0, min(SCREEN_HEIGHT, last_gaze_y + random.randint(-60, 60))))
        
        pupil_left = round(random.uniform(2.5, 3.8), 2)
        pupil_right = round(random.uniform(2.5, 3.8), 2)
        
        fixation_duration = random.randint(150, 450) if random.random() > 0.3 else 0
        saccade_amplitude = round(math.sqrt((gaze_x - last_gaze_x)**2 + (gaze_y - last_gaze_y)**2), 2)
        
        # --- 🔴 บันทึกข้อมูลลงตาราง CSV ---
        csv_writer.writerow([
            timestamp, gaze_x, gaze_y, pupil_left, pupil_right, fixation_duration, saccade_amplitude,
            current_mouse_x, current_mouse_y, current_click_event, current_scroll_dis
        ])
        
        # แสดงผลมอนิเตอร์บนหน้าจอ
        print(
            f"📥 Gaze:({gaze_x},{gaze_y}) Pupil_L:{pupil_left} | "
            f"Mouse:({current_mouse_x},{current_mouse_y}) Click:{current_click_event} Scroll:{current_scroll_dis}", 
            end="\r"
        )
        
        last_gaze_x = gaze_x
        last_gaze_y = gaze_y
        current_scroll_dis = 0
        
        time.sleep(0.05)

except KeyboardInterrupt:
    # 6. บังคับเซฟและปิดไฟล์เมื่อกด Ctrl + C
    mouse_listener.stop()
    csv_file.close()
    print(f"\n\n🎉 [สำเร็จ] ระบบทำการแยกไฟล์และบันทึกข้อมูลลง '{filename}' เรียบร้อยแล้วครับ!")"""