import serial
import pandas as pd
import time

# Serial port settings
ser = serial.Serial("COM8", 115200)  # Enter the correct serial port

# Load dataset from CSV file
df = pd.read_csv("test_data.csv")  # Enter the path of your CSV file
df = df[
    [
        "Voltage_measured (Volts)",
        "Current_measured (Amps)",
        "Time (secs)",
        "Temperature_measured (C)",
        "state",
    ]
]
# Perform one-hot encoding on the 'state' column
df = pd.get_dummies(
    df, columns=["state"], drop_first=True
)  # Convert 'state' to one-hot encoding
print(df[:4])
# Extract 10 records of Voltage, Current, Time, and state (one-hot encoded)

output = ser.readline().decode("utf-8").strip()
print(f"Received: {output}")

while "entry" not in output:
    output = ser.readline().decode("utf-8").strip()
    print(f"Received: {output}")


for i in range(1000):
    voltage = df["Voltage_measured (Volts)"].iloc[i]
    current = df["Current_measured (Amps)"].iloc[i]
    time_sec = df["Time (secs)"].iloc[i]
    state = df["state_discharge"].iloc[i]  # حالت نیازی به نرمال‌سازی ندارد

    ser.write(f"{voltage},{current},{time_sec},{state},\n".encode())

    time.sleep(1)
    output = ser.readline().decode("utf-8", errors="ignore").strip()
    print(f"Received: {output}")

# Close the serial port
ser.close()
