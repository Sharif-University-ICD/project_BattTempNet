#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"  // Converted model file as an array
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Defining necessary variables
float voltageMeasured, currentMeasured, timeMeasured;
float state1, state2, state3;  // state as one-hot encoding
float input_data[4];  // 6 features: voltage, current, time, and 3 state features
float output_data[1]; // Predicted temperature value

// Initializing variables
tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model* model;
constexpr int tensor_arena_size = 60 * 1024;  // Required space
uint8_t tensor_arena[tensor_arena_size];

// Resolver for various operations
tflite::AllOpsResolver resolver;

// Defining Interpreter pointer
tflite::MicroInterpreter* interpreter;

void setup() {
    Serial.begin(115200);

    // Loading the model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        while (1);  // Invalid model
    }

    // Initializing the Interpreter
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);

    // Allocating memory for tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        while (1);  // Memory allocation failed
    }
}

void loop() {
    if (Serial.available() > 0) {
        // Receiving data from serial
        String receivedData = Serial.readStringUntil('\n');

        // Processing the data
        int firstComma = receivedData.indexOf(',');
        int secondComma = receivedData.indexOf(',', firstComma + 1);
        int thirdComma = receivedData.indexOf(',', secondComma + 1);
        int fourthComma = receivedData.indexOf(',', thirdComma + 1);
        if (firstComma == -1 || secondComma == -1 || thirdComma == -1 || fourthComma == -1 ) {
            return;
        }

        // Receiving voltage, current, time, and state
        voltageMeasured = receivedData.substring(0, firstComma).toFloat();
        currentMeasured = receivedData.substring(firstComma + 1, secondComma).toFloat();
        timeMeasured = receivedData.substring(secondComma + 1, thirdComma).toFloat();

        // Receiving state as one-hot
        state1 = receivedData.substring(thirdComma + 1, fourthComma).toFloat();

        // Storing input values in model array
        input_data[0] = voltageMeasured;
        input_data[1] = currentMeasured;
        input_data[2] = timeMeasured;
        input_data[3] = state1;

        // Copying input data to model input tensor
        float* model_input = interpreter->input(0)->data.f;
        memcpy(model_input, input_data, sizeof(input_data));

        if (interpreter->Invoke() != kTfLiteOk) {
            return;  // Model execution failed
        }

        // Retrieving predicted temperature
        float predictedTemperature = interpreter->output(0)->data.f[0];

        // Printing the predicted temperature and input values
        Serial.println("predicted: " + String(predictedTemperature) + 
              ", voltage: " + String(input_data[0]) + 
              ", current: " + String(input_data[1]) + 
              ", time: " + String(input_data[2]) + 
              ", state: " + String(input_data[3]));
    }

    // Serial.println(predictedTemperature);
}
