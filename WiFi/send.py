import pandas as pd
import time
import requests

ESP32_IP = "192.168.80.99"  # آدرس IP دستگاه ESP32 شما
URL = f"http://{ESP32_IP}/predict"  # دیگر نیازی به تعیین پورت نیست

# بارگذاری داده‌ها از فایل CSV
df = pd.read_csv("test_data.csv")
df = df[["Voltage_measured (Volts)", "Current_measured (Amps)", "Time (secs)", "state"]]
df = pd.get_dummies(df, columns=["state"], drop_first=True)  # تبدیل state به one-hot

for i in range(len(df)):
    voltage = df.iloc[i, 0]
    current = df.iloc[i, 1]
    time_sec = df.iloc[i, 2]
    state = df.iloc[i, 3].astype(float)  # مقدار state بدون One-Hot Encoding

    params = {"voltage": voltage, "current": current, "time": time_sec, "state": state}

    try:
        response = requests.get(URL, params=params)
        print(f"Sent data: {params}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)
