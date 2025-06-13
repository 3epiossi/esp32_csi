extern "C" {
#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tfModel.h"

static QueueHandle_t queue;
#define STACK_SIZE 4 * 1024
#define DATA_COLLECTION_MODE

#ifdef DATA_COLLECTION_MODE
    #define WINDOW_SIZE 1
    #define CHANNEL_NUM (1 + 52 + 56)
    #define LLTF_INTERVAL 2
    #define HT_LFT_INTERVAL 2
#else
    #define RSSI_ONLY
    #define WINDOW_SIZE TF_NUM_INPUTS_TIMESTEP
    #define LLTF 0
    #define HT_LFT 0
    #define CHANNEL_NUM TF_NUM_INPUTS_SUBCARRIER
    #define LLTF_INTERVAL ((26/LLTF+1) << 1)
    #define HT_LFT_INTERVAL ((28/HT_LFT+1) << 1)
    #define TENSOR_ARENA_SIZE (100 * 1024)
    static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
    namespace {
        const tflite::Model* model = nullptr;
        tflite::MicroInterpreter* interpreter = nullptr;
        tflite::MicroMutableOpResolver<TF_NUM_OPS>* resolver = nullptr;
    }
    char* prediction[TF_NUM_OUTPUTS] = {"empty","metal","plastic"};
#endif

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    #define CONFIG_WIFI_BAND_MODE   WIFI_BAND_MODE_2G_ONLY
    #define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT40
    #define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT40
    #define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
    #define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
    #define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT40
#else
    #define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#endif
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   0
#define CSI_FORCE_LLTF                      0   
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
#define CONFIG_GAIN_CONTROL                 1
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
    wifi_bandwidth_t bandwidth1;
    esp_wifi_get_bandwidth(WIFI_IF_STA, &bandwidth1);
#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32S3 
    // 修正：使用正確的 wifi_interface_t 類型
    ESP_ERROR_CHECK(esp_wifi_config_espnow_rate(WIFI_IF_STA, CONFIG_ESP_NOW_RATE));
#endif
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20)
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    else
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;
    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) return;
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    BaseType_t xhigherprioritytaskwoken = pdFALSE;
    float* data = new float[CHANNEL_NUM];
    data[0] = -(float)rx_ctrl->rssi;
    uint32_t i = 1;
    #ifndef RSSI_ONLY
    for(uint32_t j = 12; j < 64; j += LLTF_INTERVAL, ++i){
        float img = (info->buf[j]);
        float rel = (info->buf[j+1]);
        data[i] = sqrt(img * img + rel * rel);
    }
    for(uint32_t j = 66; j < 118; j += LLTF_INTERVAL, ++i){
        float img = (info->buf[j]);
        float rel = (info->buf[j+1]);
        data[i] = sqrt(img * img + rel * rel);
    }
    for(uint32_t j = 132; j < 188; j += HT_LFT_INTERVAL, ++i){
        float img = (info->buf[j]);
        float rel = (info->buf[j+1]);
        data[i] = sqrt(img * img + rel * rel);
    }
    for(uint32_t j = 190; j < 246; j += HT_LFT_INTERVAL, ++i){
        float img = (info->buf[j]);
        float rel = (info->buf[j+1]);
        data[i] = sqrt(img * img + rel * rel);
    }
    #endif
    if (i != CHANNEL_NUM) {
        ets_printf("Error: i=%u, CHANNEL_NUM=%u\n", i, CHANNEL_NUM);
    }
    
    if (xQueueSendFromISR(queue, (void*)&data, &xhigherprioritytaskwoken) != pdTRUE) {
        float* dummy;
        xQueueReceiveFromISR(queue, (void*)&dummy, &xhigherprioritytaskwoken);
        delete[] dummy;
        xQueueSendFromISR(queue, (void*)&data, &xhigherprioritytaskwoken);
    }
    portYIELD_FROM_ISR(xhigherprioritytaskwoken);
}

#ifndef DATA_COLLECTION_MODE
bool lstm_init() {
    model = tflite::GetModel(tfModel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\n");
        return false;
    }

    static tflite::MicroMutableOpResolver<TF_NUM_OPS> static_resolver;
    resolver = &static_resolver;

    resolver->AddUnidirectionalSequenceLSTM();
    resolver->AddFullyConnected();
    resolver->AddSoftmax();
    resolver->AddTanh();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddReshape();
    resolver->AddUnpack();
    resolver->AddLogistic();
    resolver->AddShape();
    resolver->AddStridedSlice();
    resolver->AddTranspose();
    resolver->AddPack();
    resolver->AddFill();
    resolver->AddSplit();

    static tflite::MicroInterpreter static_interpreter(
        model, *resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return false;
    }

    return true;
}

void normalize_input(float* input, int len) {
    for (int i = 0; i < len; ++i) {
        input[i] = (input[i] - CSI_MEAN) / (CSI_STD + 1e-8f);
    }
}

bool predict_lstm(float* input_data, int timestep, int subcarrier) {
    if (!interpreter) {
        printf("Interpreter not initialized. Call lstm_init() first.\n");
        return false;
    }

    TfLiteTensor* input = interpreter->input(0);
    if (input->type != kTfLiteFloat32) {
        printf("Input tensor type is not float32!\n");
        return false;
    }

    if (input->dims->size != 3 ||
        input->dims->data[0] != 1 ||
        input->dims->data[1] != timestep ||
        input->dims->data[2] != subcarrier) {
        printf("Input tensor shape mismatch! Expected [1,%d,%d], got [%d,%d,%d]\n",
               timestep, subcarrier,
               input->dims->data[0], input->dims->data[1], input->dims->data[2]);
        return false;
    }

    memcpy(input->data.f, input_data, timestep * subcarrier * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke failed\n");
        return false;
    }

    TfLiteTensor* output = interpreter->output(0);
    int index = 0;
    printf("Output: ");
    for (int i = 0; i < TF_NUM_OUTPUTS; ++i) {
        if(output->data.f[i] > output->data.f[index]) index = i;
        printf("%f ", output->data.f[i]);
    }
    printf("\n");
    printf("prediction : %s\n", prediction[index]);

    return true;
}
#endif


void taskReceive(void* pvParameters){
    float* data;
    float input[WINDOW_SIZE][CHANNEL_NUM];
    float rssi_value = 100.0;
    uint8_t trigger;
    #ifndef DATA_COLLECTION_MODE
        printf("predict starting\n");
    #endif
    for(;;){
        trigger = 0;
        for(uint32_t i = 0; i < WINDOW_SIZE;){
            if(xQueueReceive(queue, (void*)&data, portMAX_DELAY) != pdPASS){
                continue;
            }
            memcpy(input[i], data, CHANNEL_NUM * sizeof(float));
            delete[] data;
            #ifdef DATA_COLLECTION_MODE
                for(int j = 0; j < CHANNEL_NUM; ++j){
                    printf("%f ", input[i][j]);
                }
                printf("\n");
            #endif
            #ifndef DATA_COLLECTION_MODE
                if(trigger == 0 && (abs(input[i][0] - rssi_value) < 3.0)){
                    rssi_value = input[i][0];
                    continue;
                }
                else{
                    rssi_value = input[i][0];
                    trigger = 1;
                }
            #endif
            ++i;
        }
        // 這裡將 int32_t 轉 float 並呼叫 predict_lstm
        #ifndef DATA_COLLECTION_MODE
            normalize_input(&input[0][0], WINDOW_SIZE * CHANNEL_NUM);
            predict_lstm(&input[0][0], WINDOW_SIZE, CHANNEL_NUM);
        #endif
    }
}
    
static void wifi_csi_init()
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

extern "C" void app_main()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    wifi_init();
    wifi_csi_init();
    #ifndef DATA_COLLECTION_MODE
    if (!lstm_init()) {
        printf("LSTM init failed!\n");
        return;
    }
    #endif
    queue = xQueueCreate(20, sizeof(float*));
    TaskHandle_t xHandle = NULL;
    BaseType_t xReturned;
    xReturned = xTaskCreate(
                    taskReceive,       /* Function that implements the task. */
                    "taskReceive",          /* Text name for the task. */
                    STACK_SIZE,      /* Stack size in words, not bytes. */
                    NULL,    /* Parameter passed into the task. */
                    1,/* Priority at which the task is created. */
                    &xHandle );      /* Used to pass out the created task's handle. */
    // run_lstm(idle, TF_NUM_INPUTS);
}