import cv2
import serial
import time
import torch
import numpy as np
from ultralytics import YOLO

# --- PENGATURAN ROBOT ---
# Sesuaikan ini sesuai setup Anda.
SERIAL_PORT = 'COM5'          # Dengan port serial ESP32
PRIMARY_TARGET = 'cell phone' # Objek yang diikuti robot
AVOID_TARGET = 'chair'        # Objek yang dihindari robot
DISTANCE_THRESHOLD = 5.0      # Ambang batas jarak (lebih tinggi = lebih dekat)
REVERSE_WAIT_TIME = 2.0       # Waktu mundur jika robot terjebak (detik)
# -------------------------

# Coba hubungkan ke ESP32
try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    print(f"✅ Terhubung ke ESP32 di {SERIAL_PORT}")
    time.sleep(2)
except serial.SerialException:
    print(f"❌ Gagal terhubung ke port {SERIAL_PORT}. Cek kabel atau program lain.")
    exit()

# Muat model AI (YOLO untuk deteksi, MiDaS untuk jarak)
print("🧠 Memuat model AI...")
try:
    model_yolo = YOLO('yolov8n.pt')
    model_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    print("✅ Model AI siap.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}. Periksa internet atau instalasi.")
    exit()

# Buka webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Gagal membuka webcam.")
    ser.close()
    exit()

print(f"\n🚀 Robot siap! Ikuti '{PRIMARY_TARGET}', Hindari '{AVOID_TARGET}'.")
last_command, is_stuck, stuck_timestamp = '', False, 0

# --- LOOP UTAMA ROBOT ---
while True:
    ret, frame = cap.read()
    if not ret: break # Keluar jika gagal baca frame

    frame_height, frame_width, _ = frame.shape

    # Deteksi objek & estimasi jarak
    img_transformed = transform(frame).to(device)
    with torch.no_grad():
        prediction = model_midas(img_transformed)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth_map = prediction.cpu().numpy()
    results_yolo = model_yolo(frame, verbose=False)
    annotated_frame = results_yolo[0].plot()

    avoid_target_spotted, primary_target_spotted = False, False
    target_x_center, target_avg_depth = 0, 0

    # Proses hasil deteksi
    for r in results_yolo:
        for box in r.boxes:
            class_name = model_yolo.names[int(box.cls[0])]
            
            if class_name == AVOID_TARGET:
                avoid_target_spotted = True
            elif class_name == PRIMARY_TARGET:
                primary_target_spotted = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                target_x_center = int((x1 + x2) / 2)
                roi_depth = depth_map[y1:y2, x1:x2]
                target_avg_depth = np.mean(roi_depth) if roi_depth.size > 0 else 0
                cv2.putText(annotated_frame, f"Jarak: {target_avg_depth:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Logika kendali pergerakan robot
    command = '0' # Default: STOP
    if avoid_target_spotted:
        command, is_stuck = '0', False # Berhenti jika ada objek yang harus dihindari
        cv2.putText(annotated_frame, f"STOP: {AVOID_TARGET}!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif primary_target_spotted:
        # Jika target utama terlihat:
        if target_x_center < frame_width / 3:
            command = '3' # Belok Kiri
            is_stuck = False
        elif target_x_center > frame_width * 2 / 3:
            command = '2' # Belok Kanan
            is_stuck = False
        else: # Target sudah di tengah, atur maju/mundur berdasarkan jarak
            if target_avg_depth > DISTANCE_THRESHOLD:
                if not is_stuck: # Baru mendekati ambang batas
                    is_stuck, stuck_timestamp, command = True, time.time(), '0'
                else: # Sudah terlalu dekat cukup lama, mundur
                    if time.time() - stuck_timestamp > REVERSE_WAIT_TIME:
                        command = '4' # Mundur
                    else:
                        command = '0' # Tahan (tetap berhenti)
            else:
                is_stuck = False
                command = '1' # Maju
    else: # Tidak ada target atau penghalang
        is_stuck = False
        command = '0' # Berhenti

    # Kirim perintah ke ESP32 (hanya jika ada perubahan perintah)
    if command != last_command:
        ser.write(command.encode('utf-8'))
        print(f"Status: Hindari='{avoid_target_spotted}', Ikuti='{primary_target_spotted}' -> Perintah: {command}")
        last_command = command

    # Tampilkan video dan cek tombol keluar
    cv2.imshow("Otak AI Robot - Tekan 'q' untuk keluar", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- SHUTDOWN ---
print("Mematikan sistem...")
ser.write(b'0') # Pastikan robot berhenti total
ser.close()
cap.release()
cv2.destroyAllWindows()
print("Sistem mati.")