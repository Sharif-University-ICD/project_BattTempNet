import serial
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Serial port settings
ser = serial.Serial("COM8", 115200, timeout=1)  # Set timeout to prevent freezing

# Load dataset from CSV file
df = pd.read_csv("test_data.csv")
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
df = pd.get_dummies(df, columns=["state"], drop_first=True)

# Wait for the start message from the serial port
while True:
    output = ser.readline().decode("utf-8", errors="ignore").strip()
    print(f"Received: {output}")
    if "entry" in output:
        break

start_time = time.time()  # Start timing

actual_temperatures = []  # Store true temperature values from dataset
predicted_temperatures = []  # Store received temperature values from serial

# Send data
for i in range(1000):
    voltage = df["Voltage_measured (Volts)"].iloc[i]
    current = df["Current_measured (Amps)"].iloc[i]
    time_sec = df["Time (secs)"].iloc[i]
    state = df["state_discharge"].iloc[i]
    true_temp = df["Temperature_measured (C)"].iloc[i]  # Actual temperature

    data = f"{voltage},{current},{time_sec},{state},\n"
    ser.write(data.encode())

    # Read from serial without excessive delay
    start_read = time.time()
    while ser.in_waiting == 0 and (time.time() - start_read) < 0.1:
        pass
    output = ser.readline().decode("utf-8", errors="ignore").strip()
    print(f"Received: {output}")

    try:
        predicted_temp = float(output)  # Convert received string to float
        actual_temperatures.append(true_temp)
        predicted_temperatures.append(predicted_temp)
    except ValueError:
        print(f"Invalid temperature received: {output}")  # Debugging message

end_time = time.time()  # End timing

# Calculate total transfer time
transfer_time = end_time - start_time
print(f"Total transfer time: {transfer_time:.2f} seconds")

# Close the serial port
ser.close()

# Convert lists to numpy arrays
actual_temperatures = np.array(actual_temperatures)
predicted_temperatures = np.array(predicted_temperatures)

# Ensure we have valid data before computing MSE
if len(actual_temperatures) == 0 or len(predicted_temperatures) == 0:
    print("Error: No valid data received. Cannot compute MSE.")
else:
    mse = mean_squared_error(actual_temperatures, predicted_temperatures)
    print(f"Mean Squared Error (MSE): {mse:.5f}")

    # Plot actual vs predicted temperature
    plt.figure(figsize=(10, 5))
    plt.plot(actual_temperatures, label="Actual Temperature", linestyle="--")
    plt.plot(predicted_temperatures, label="Predicted Temperature", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.title("Comparison of Actual vs Predicted Temperature")
    plt.show()
