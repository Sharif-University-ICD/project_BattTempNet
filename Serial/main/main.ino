#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"  // فایل مدل تبدیل‌شده به آرایه
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// تعریف متغیرهای مورد نیاز
float voltageMeasured, currentMeasured, timeMeasured;
float state1, state2, state3;  // state به صورت one-hot encoding
float input_data[4];  // 6 ویژگی: ولتاژ، جریان، زمان و 3 ویژگی state
float output_data[1]; // مقدار پیش‌بینی‌شده دما

// تعریف مقداردهی اولیه
tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model* model;
constexpr int tensor_arena_size = 60 * 1024;  // فضای مورد نیاز
uint8_t tensor_arena[tensor_arena_size];

// resolver برای عملیات‌های مختلف
tflite::AllOpsResolver resolver;

// تعریف اشاره‌گر Interpreter
tflite::MicroInterpreter* interpreter;

void setup() {
    Serial.begin(115200);

    // بارگذاری مدل
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        while (1);  // مدل نادرست است
    }

    // مقداردهی اولیه Interpreter
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);

    // تخصیص حافظه به تنسورها
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        while (1);  // تخصیص حافظه انجام نشد
    }
}

void loop() {
    if (Serial.available() > 0) {
        // دریافت داده‌ها از سریال
        String receivedData = Serial.readStringUntil('\n');

        // پردازش داده‌ها
        int firstComma = receivedData.indexOf(',');
        int secondComma = receivedData.indexOf(',', firstComma + 1);
        int thirdComma = receivedData.indexOf(',', secondComma + 1);
        int fourthComma = receivedData.indexOf(',', thirdComma + 1);
        if (firstComma == -1 || secondComma == -1 || thirdComma == -1 || fourthComma == -1 ) {
            return;
        }

        // دریافت ولتاژ، جریان، زمان و state
        voltageMeasured = receivedData.substring(0, firstComma).toFloat();
        currentMeasured = receivedData.substring(firstComma + 1, secondComma).toFloat();
        timeMeasured = receivedData.substring(secondComma + 1, thirdComma).toFloat();

        // دریافت state به صورت one-hot
        state1 = receivedData.substring(thirdComma + 1, fourthComma).toFloat();

        // قرار دادن مقادیر ورودی در آرایه مدل
        input_data[0] = voltageMeasured;
        input_data[1] = currentMeasured;
        input_data[2] = timeMeasured;
        input_data[3] = state1;


        float* model_input = interpreter->input(0)->data.f;
        memcpy(model_input, input_data, sizeof(input_data));

        if (interpreter->Invoke() != kTfLiteOk) {
            return;  // اجرای مدل انجام نشد
        }


      float predictedTemperature = interpreter->output(0)->data.f[0];

    Serial.println("predicted: " + String(predictedTemperature) + 
              ", voltage: " + String(input_data[0]) + 
              ", current: " + String(input_data[1]) + 
              ", time: " + String(input_data[2]) + 
              ", state: " + String(input_data[3]));
    }

    //Serial.println(predictedTemperature);
          
}
