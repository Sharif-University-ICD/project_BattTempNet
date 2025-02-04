import pandas as pd
import time
import requests

ESP32_IP = "192.168.80.99"  # IP address of your ESP32 device
URL = f"http://{ESP32_IP}/predict"  # No need to specify the port anymore

# Load data from the CSV file
df = pd.read_csv("test_data.csv")
df = df[["Voltage_measured (Volts)", "Current_measured (Amps)", "Time (secs)", "state"]]
df = pd.get_dummies(
    df, columns=["state"], drop_first=True
)  # Convert state to one-hot encoding

for i in range(len(df)):
    voltage = df.iloc[i, 0]
    current = df.iloc[i, 1]
    time_sec = df.iloc[i, 2]
    state = df.iloc[i, 3].astype(float)  # state value without one-hot encoding

    params = {"voltage": voltage, "current": current, "time": time_sec, "state": state}

    try:
        response = requests.get(URL, params=params)
        print(f"Sent data: {params}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)  # Delay between requests
