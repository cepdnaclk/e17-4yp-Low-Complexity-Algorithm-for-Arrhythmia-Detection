/* UART Echo Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "sdkconfig.h"
#include "esp_log.h"
#include <stdlib.h> 



typedef int boolean;
#define true  1
#define false 0

#define M 				5
#define N 				30
#define winSize			250
#define HP_CONSTANT		((float) 1 / (float) M)

/**
 * This is an example which echos any data it receives on configured UART back to the sender,
 * with hardware flow control turned off. It does not use UART driver event queue.
 *
 * - Port: configured UART
 * - Receive (Rx) buffer: on
 * - Transmit (Tx) buffer: off
 * - Flow control: off
 * - Event queue: off
 * - Pin assignment: see defines below (See Kconfig)
 */

#define ECHO_TEST_TXD GPIO_NUM_1
#define ECHO_TEST_RXD GPIO_NUM_3
#define ECHO_TEST_RTS (UART_PIN_NO_CHANGE)
#define ECHO_TEST_CTS (UART_PIN_NO_CHANGE)

#define ECHO_UART_PORT_NUM      UART_NUM_0
#define ECHO_UART_BAUD_RATE     115200
#define ECHO_TASK_STACK_SIZE    2048

#define BUF_SIZE (1024)

static void echo_task(void *arg)
{

    // circular buffer for input ecg signal
	// we need to keep a history of M + 1 samples for HP filter
	float ecg_buff[M + 1] = {0};
	int ecg_buff_WR_idx = 0;
	int ecg_buff_RD_idx = 0;
	
	// circular buffer for input ecg signal
	// we need to keep a history of N+1 samples for LP filter
	float hp_buff[N + 1] = {0};
	int hp_buff_WR_idx = 0;
	int hp_buff_RD_idx = 0;
	
	// LP filter outputs a single point for every input point
	// This goes straight to adaptive filtering for eval
	float next_eval_pt = 0;
	
	// running sums for HP and LP filters, values shifted in FILO
	float hp_sum = 0;
	float lp_sum = 0;
	
	// parameters for adaptive thresholding
	double treshold = 0;
	boolean triggered = false;
	int trig_time = 0;
	float win_max = 0;
	int win_idx = 0;
	
	int i = 0;


    /* Configure parameters of an UART driver,
     * communication pins and install the driver */
    uart_config_t uart_config = {
        .baud_rate = ECHO_UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB,
    };
    int intr_alloc_flags = 0;

#if CONFIG_UART_ISR_IN_IRAM
    intr_alloc_flags = ESP_INTR_FLAG_IRAM;
#endif

    ESP_ERROR_CHECK(uart_driver_install(ECHO_UART_PORT_NUM, BUF_SIZE * 2, 0, 0, NULL, intr_alloc_flags));
    ESP_ERROR_CHECK(uart_param_config(ECHO_UART_PORT_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(ECHO_UART_PORT_NUM, ECHO_TEST_TXD, ECHO_TEST_RXD, ECHO_TEST_RTS, ECHO_TEST_CTS));

    // Configure a temporary buffer for the incoming data
    // uint8_t *data = (uint8_t *) malloc(BUF_SIZE);
    float *data = (float *) malloc(BUF_SIZE);

    while(1) {
        // Read data from the UART
        int len = uart_read_bytes(ECHO_UART_PORT_NUM, data, BUF_SIZE, 20 / portTICK_PERIOD_MS);
        vTaskDelay(2000 / portTICK_PERIOD_MS);
        if (len > 0){
            
            ///convert received data to a float
            float value = atof((const char *)data);

            // char buffer[20];
            
            // // Convert float to string
            // sprintf(buffer, "%.5f", value);

            // ESP_LOGI("READING ", "Value received %s bytes ",buffer);
            // ESP_LOGI("READING ", "Wrote %s bytes len %d",(const char *)data, len);
            
            ecg_buff[ecg_buff_WR_idx++] = value;
            ecg_buff_WR_idx %= (M+1);
            
            printf("i - %d\n", i);
            
            /* High pass filtering */
            if(i < M){
                // first fill buffer with enough points for HP filter
                hp_sum += ecg_buff[ecg_buff_RD_idx];
                hp_buff[hp_buff_WR_idx] = 0;
                
                //printf("hp_buff[hp_buff_WR_idx] - %f\n", hp_buff[hp_buff_WR_idx]);
            }
            else{
                hp_sum += ecg_buff[ecg_buff_RD_idx];
                
                int tmp = ecg_buff_RD_idx - M;
                if(tmp < 0) tmp += M + 1;
                
                hp_sum -= ecg_buff[tmp];
                
                float y1 = 0;
                float y2 = 0;
                
                tmp = (ecg_buff_RD_idx - ((M+1)/2));
                if(tmp < 0) tmp += M + 1;
                
                y2 = ecg_buff[tmp];
                
                y1 = HP_CONSTANT * hp_sum;
                
                hp_buff[hp_buff_WR_idx] = y2 - y1;
                
                //printf("hp_buff[hp_buff_WR_idx] - %f\n", hp_buff[hp_buff_WR_idx]);
            }
            
            // done reading ECG buffer, increment position
            ecg_buff_RD_idx++;
            ecg_buff_RD_idx %= (M+1);
            
            // done writing to HP buffer, increment position
            hp_buff_WR_idx++;
            hp_buff_WR_idx %= (N+1);
            
            /* Low pass filtering */
            
            // shift in new sample from high pass filter
            lp_sum += hp_buff[hp_buff_RD_idx] * hp_buff[hp_buff_RD_idx];
            
            if(i < N){
                // first fill buffer with enough points for LP filter
                next_eval_pt = 0;
                
            }
            else{
                // shift out oldest data point
                int tmp = hp_buff_RD_idx - N;
                if(tmp < 0) tmp += N+1;
                
                lp_sum -= hp_buff[tmp] * hp_buff[tmp];
                
                next_eval_pt = lp_sum;
            }
            
            // done reading HP buffer, increment position
            hp_buff_RD_idx++;
            hp_buff_RD_idx %= (N+1);
            
            /* Adapative thresholding beat detection */
            // set initial threshold				
            if(i < winSize) {
                if(next_eval_pt > treshold) {
                    treshold = next_eval_pt;
                }
            }
            
            // check if detection hold off period has passed
            if(triggered){
                trig_time++;
                
                if(trig_time >= 100){
                    triggered = false;
                    trig_time = 0;
                }
            }
            
            // find if we have a new max
            if(next_eval_pt > win_max) win_max = next_eval_pt;
            
            // find if we are above adaptive threshold
            if(next_eval_pt > treshold && !triggered) {
                // result[i] = 1;
                ESP_LOGI("result"," 1"); 
                triggered = true;
            }
            else {
                // result[i] = 0;
                ESP_LOGI("result"," 0"); 
            }
		
            // adjust adaptive threshold using max of signal found 
            // in previous window            
            if(win_idx++ >= winSize){
                // weighting factor for determining the contribution of
                // the current peak value to the threshold adjustment
                double gamma = 0.175;
                
                // forgetting factor - 
                // rate at which we forget old observations
                double alpha = 0.01 + ( ((float) rand() / (float) RAND_MAX) * ((0.1 - 0.01)));
                
                treshold = alpha * gamma * win_max + (1 - alpha) * treshold;
                
                // reset current window ind
                win_idx = 0;
                win_max = -10000000;
            }
            i++;









            
           // ESP_LOGI("first byte %s", value);
             // Write data back to the UART
            uart_write_bytes(ECHO_UART_PORT_NUM, (const char *) data, len);
            
        }
        
       
       // vTaskDelay(2000 / portTICK_PERIOD_MS);
    }
}

void app_main(void)
{
    xTaskCreate(echo_task, "uart_echo_task", ECHO_TASK_STACK_SIZE, NULL, 10, NULL);
}

// /* UART asynchronous example, that uses separate RX and TX tasks

//    This example code is in the Public Domain (or CC0 licensed, at your option.)

//    Unless required by applicable law or agreed to in writing, this
//    software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//    CONDITIONS OF ANY KIND, either express or implied.
// */
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_system.h"
// #include "esp_log.h"
// #include "driver/uart.h"
// #include "string.h"
// #include "driver/gpio.h"

// static const int RX_BUF_SIZE = 1024;

// #define TXD_PIN (GPIO_NUM_1)
// #define RXD_PIN (GPIO_NUM_3)

// void init(void) {
//     const uart_config_t uart_config = {
//         .baud_rate = 115200,
//         .data_bits = UART_DATA_8_BITS,
//         .parity = UART_PARITY_DISABLE,
//         .stop_bits = UART_STOP_BITS_1,
//         .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
//         .source_clk = UART_SCLK_APB,
//     };
//     // We won't use a buffer for sending data.
//     uart_driver_install(UART_NUM_1, RX_BUF_SIZE * 2, 0, 0, NULL, 0);
//     uart_param_config(UART_NUM_1, &uart_config);
//     uart_set_pin(UART_NUM_1, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
// }

// int sendData(const char* logName, const char* data)
// {
//     const int len = strlen(data);
//     const int txBytes = uart_write_bytes(UART_NUM_0, data, len);
//     ESP_LOGI(logName, "Wrote %d bytes", txBytes);
//     return txBytes;
// }

// static void tx_task(void *arg)
// {
//     static const char *TX_TASK_TAG = "TX_TASK";
//     esp_log_level_set(TX_TASK_TAG, ESP_LOG_INFO);
//     while (1) {
//         sendData(TX_TASK_TAG, "Hello world");
//         vTaskDelay(2000 / portTICK_PERIOD_MS);
//     }
// }

// static void rx_task(void *arg)
// {
//     static const char *RX_TASK_TAG = "RX_TASK";
//     esp_log_level_set(RX_TASK_TAG, ESP_LOG_INFO);
//     uint8_t* data = (uint8_t*) malloc(RX_BUF_SIZE+1);
//     while (1) {
//         const int rxBytes = uart_read_bytes(UART_NUM_0, data, RX_BUF_SIZE, 1000 / portTICK_PERIOD_MS);
//         if (rxBytes > 0) {
//             data[rxBytes] = 0;
//             ESP_LOGI(RX_TASK_TAG, "Read %d bytes: '%s'", rxBytes, data);
//             ESP_LOG_BUFFER_HEXDUMP(RX_TASK_TAG, data, rxBytes, ESP_LOG_INFO);
//         }
//     }
//     free(data);
// }

// void app_main(void)
// {
//     init();
//     xTaskCreate(rx_task, "uart_rx_task", 1024*2, NULL, configMAX_PRIORITIES, NULL);
//     xTaskCreate(tx_task, "uart_tx_task", 1024*2, NULL, configMAX_PRIORITIES-1, NULL);
// }