import pandas as pd
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

ESP32_IP = "192.168.80.99"  # IP address of ESP32 device
URL = f"http://{ESP32_IP}/predict"

# Load data from CSV file
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
df = pd.get_dummies(df, columns=["state"], drop_first=True)

actual_temperatures = []  # Store actual temperature values
predicted_temperatures = []  # Store predicted temperature values

start_time = time.time()  # Start timing

for i in range(1000):
    voltage = df.iloc[i, 0]
    current = df.iloc[i, 1]
    time_sec = df.iloc[i, 2]
    state = df.iloc[i, 4].astype(float)  # Extract state value
    true_temp = df.iloc[i, 3]  # Extract actual temperature

    params = {"voltage": voltage, "current": current, "time": time_sec, "state": state}

    try:
        response = requests.get(URL, params=params, timeout=0.2)  # Reduce timeout
        predicted_temp = float(response.text.strip())  # Convert response to float

        actual_temperatures.append(true_temp)
        predicted_temperatures.append(predicted_temp)

        # print(f"Sent: {params}")
        print(f"Received: {predicted_temp}")
    except Exception as e:
        print(f"Error: {e}")

end_time = time.time()  # End timing

# Compute total time
transfer_time = end_time - start_time
print(f"Total transfer time: {transfer_time:.2f} seconds")

# Convert lists to numpy arrays
actual_temperatures = np.array(actual_temperatures)
predicted_temperatures = np.array(predicted_temperatures)

# Compute MSE
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
