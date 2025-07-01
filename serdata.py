import serial
import time
import csv
from datetime import datetime
import os

LOG_FILE = 'manual_sensor_log.csv'

try:
    ser = serial.Serial('COM7', 115200, timeout=2)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"🔌 Connected to {ser.name}")

    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        with open(LOG_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'sys_bp', 'dia_bp', 'heart_rate', 'spo2'])

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print("🔄 Serial Line Received:", repr(line))

            if "HR:" in line and "SpO2:" in line and "BP:" in line:
                try:
                    parts = line.split(',')

                    hr = float(parts[0].split(':')[1].strip().split()[0])
                    spo2 = float(parts[1].split(':')[1].strip().replace('%', ''))
                    bp_raw = parts[2].split(':')[1].strip()

                    if "N/A" in bp_raw:
                        sys, dia = 0.0, 0.0
                    else:
                        sys = float(bp_raw.split(',')[0]) if ',' in bp_raw else float(bp_raw.split()[0])
                        dia = float(parts[3]) if len(parts) > 3 else 0.0

                    with open(LOG_FILE, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([datetime.now(), sys, dia, hr, spo2])
                        print("✅ Logged:", sys, dia, hr, spo2)

                except Exception as e:
                    print("⚠️ Parse Error:", e)
        else:
            print("⚠️ No data received")

        time.sleep(1)

except serial.SerialException as e:
    print("❌ Serial error:", e)

except KeyboardInterrupt:
    print("\n👋 Exiting...")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("🔒 Serial connection closed.")
