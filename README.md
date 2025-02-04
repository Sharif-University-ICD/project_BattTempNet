# Battery Temperature Prediction with ESP32

## Project Summary

This project involves the implementation of a neural network model on an ESP32 microcontroller to predict battery temperature based on voltage, current, time, and state data. The system uses TensorFlow Lite for microcontrollers (TFLite Micro) to run the model, which was trained using data collected in previous experiments. The ESP32 is connected to a Wi-Fi network, allowing it to receive data via HTTP requests and provide real-time temperature predictions.

## Team Members

- **Mitra Ghali Pour** (Student ID: 401106363)
- **Melika Alizadeh** (Student ID: 401106255)
- **Elina Hojabari** (Student ID: 401170661)

## How It Works

1. **Model Deployment on ESP32**: 
   - The trained TensorFlow Lite model is converted into a C array and deployed onto the ESP32.
   - The ESP32 runs the model to predict battery temperature using input features: voltage, current, time, and state.

2. **Wi-Fi Connection**: 
   - The ESP32 connects to a Wi-Fi network to receive data from a web server.
   
3. **Prediction via HTTP**:
   - Data is sent to the ESP32 over HTTP requests containing the input values (voltage, current, time, and state).
   - The ESP32 processes the data, runs the model inference, and sends back the predicted temperature.

## Requirements

- ESP32 microcontroller
- TensorFlow Lite for Microcontrollers (TFLite Micro)
- Wi-Fi network for communication

## How to Use

1. Flash the provided code to your ESP32.
2. Ensure that the ESP32 is connected to a Wi-Fi network.
3. Use an HTTP client (e.g., a Python script or browser) to send data (voltage, current, time, state) to the ESP32.
4. The ESP32 will return the predicted temperature based on the input data.
