import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Masking, Input, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load the data
data = []
for file in sorted(os.listdir("./csv/")):
    if "_charge" in file:
        a = pd.read_csv(os.path.join("./csv/", file))
        if a.isnull().values.any():
            continue
        a = a.to_numpy()
        data.append(a[10:1000])
        
data = pad_sequences(data, padding='post', dtype='float32')

# Separate input (voltage, current, time) and output (temperature)
X = data[:, :, (0, 1, 3, 4, 5)]  # Features: voltage, current, charge, time
y = data[:, :, 2:3]  # Battery temperature

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a CNN model considering temporal dependencies
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))

# 1D Convolutional layers with Causal Padding
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='causal'))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal'))

# Layer normalization and dropout for improved performance
model.add(LayerNormalization())
model.add(Dropout(0.2))

# Output layer with time-sequential structure
model.add(Dense(1))  # Output preserves the temporal structure of the input

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
model.summary()

# Train the model
history = model.fit(
    X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val)
)

# Evaluate the model
loss = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")

# Predict for a sequence of data
predicted_temperature = model.predict(X_val[:1])
print(f"Predicted Temperature: {predicted_temperature.shape}")
print(f"True Temperature: {y_val[0, -3:, 0]}")

# Save the model
model.save("cnn_temporal_model.h5")