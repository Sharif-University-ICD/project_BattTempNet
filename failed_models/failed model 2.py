
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = []
count_battery = 6
for i in range(count_battery):
    for folder in sorted(os.listdir("./dataset/" + str(i + 1) +"/")):
        if not folder.startswith("B0"):
            continue

        root = os.path.join('dataset', str(i + 1), folder, 'csv')
        for file in sorted(os.listdir(root)):
            if '_charge' in file:
                a = pd.read_csv(os.path.join(root, file))
                if a.isnull().values.any():
                    continue
                a = a.to_numpy()
                data.append(a[10:1000])


data = pad_sequences(data, padding='post', dtype='float32')

print(data.shape)

# Split the data into input (voltage, current) and target (temperature)
X = data[:, :,(0,1,3,4)]  # voltage and current
y = data[:, :, 2:3]  # temperature
# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train[0,:5,:])

# Define the LSTM model
model = Sequential()

# Masking layer to handle varying lengths of time series (if any)
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))

# LSTM layer
model.add(LSTM(5, activation="relu", return_sequences=True))

# Dense layer to predict temperature
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Print the model summary
model.summary()

print(X_train.shape , y_train.shape)

# Train the model
history = model.fit(
    X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val)
)

# Evaluate the model
loss = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
# New voltage and current data
predicted_temperature = model.predict(X_val[:1])
print(predicted_temperature.shape)
print(f"Predicted Temperature: {predicted_temperature[0,-3:,0]}")
print(f"True Temperature: {y_val[0,-3:,0]}")

model.save_weights("./model_weights_charge.tf")
# model.load_weights("./model_weights_charge.tf")