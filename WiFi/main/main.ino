#include <WiFi.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"  // فایل مدل تبدیل‌شده به آرایه
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// تعریف متغیرهای مورد نیاز
float input_data[4];  // 4 ویژگی: ولتاژ، جریان، زمان و state
float output_data[1]; // مقدار پیش‌بینی‌شده دما

// تعریف مقداردهی اولیه
constexpr int tensor_arena_size = 60 * 1024;  // فضای مورد نیاز
uint8_t tensor_arena[tensor_arena_size];
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter;
const tflite::Model* model;

// اطلاعات شبکه WiFi
const char* ssid = "Mi";
const char* password = "12345678";
WiFiServer server(80);

void setup() {
    Serial.begin(115200);
    delay(1000);

    // اتصال به WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // بارگذاری مدل
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        while (1);
    }

    // مقداردهی اولیه Interpreter
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        while (1);
    }

    server.begin();
}

float inference(float voltage, float current, float time, float state) {
    input_data[0] = voltage;
    input_data[1] = current;
    input_data[2] = time;
    input_data[3] = state;

    memcpy(interpreter->input(0)->data.f, input_data, sizeof(input_data));
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Model execution failed");
        return -1;
    }

    return interpreter->output(0)->data.f[0];
}

void handleGetRequest(WiFiClient client, String request) {
    if (request.indexOf("/predict?") != -1) {
        int vIndex = request.indexOf("voltage=") + 8;
        int cIndex = request.indexOf("current=") + 8;
        int tIndex = request.indexOf("time=") + 5;
        int sIndex = request.indexOf("state=") + 6;

        if (vIndex == 7 || cIndex == 7 || tIndex == 4 || sIndex == 5) {
            client.println("HTTP/1.1 400 Bad Request");
            client.println("Content-Type: text/plain");
            client.println();
            client.println("Invalid parameters");
            client.stop();
            return;
        }

        float voltage = request.substring(vIndex, request.indexOf('&', vIndex)).toFloat();
        float current = request.substring(cIndex, request.indexOf('&', cIndex)).toFloat();
        float time = request.substring(tIndex, request.indexOf('&', tIndex)).toFloat();
        float state = request.substring(sIndex, request.indexOf(' ', sIndex)).toFloat();

        float predictedTemperature = inference(voltage, current, time, state);

        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: text/plain");
        client.println();
        client.println(String(predictedTemperature));
    }
}

void loop() {
    WiFiClient client = server.available();
    if (client) {
        String request = client.readStringUntil('\r');
        Serial.println("Received request: " + request);
        handleGetRequest(client, request);
        client.stop();
    }
}
