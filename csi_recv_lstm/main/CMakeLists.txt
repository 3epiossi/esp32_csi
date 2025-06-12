set(SRCS "app_main.cpp")
set(INCLUDE_DIRS 
    "."
)

idf_component_register(SRCS ${SRCS}
                       INCLUDE_DIRS ${INCLUDE_DIRS}
                       REQUIRES espressif__esp-tflite-micro nvs_flash esp_wifi esp_netif freertos
                       )
