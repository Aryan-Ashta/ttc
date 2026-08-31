/*
 * Phase 3 hardware test: runs model_run() against the same 500 test
 * vectors validated on host
 * Prints a PASS/FAIL summary over UART (GPIO0=TX, GPIO1=RX at 115200
 * baud) and blinks the onboard LED (GPIO25, standard Pico)
 * solid on PASS, fast-blink on FAIL.
 */
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "model.h"
#include "test_vectors.h"

#define LED_PIN 25

static void blink_result(int passed) {
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    if (passed) {
        gpio_put(LED_PIN, 1);  /* solid on = PASS */
    } else {
        while (1) {            /* fast blink forever = FAIL */
            gpio_put(LED_PIN, 1);
            sleep_ms(100);
            gpio_put(LED_PIN, 0);
            sleep_ms(100);
        }
    }
}

int main(void) {
    stdio_init_all();
    sleep_ms(2000);  /* give a serial terminal time to attach */

    printf("\n--- TTC Phase 3: model_run() on RP2040 ---\n");
    printf("arena=%d bytes, input=%d bytes, output=%d bytes\n",
           MODEL_ARENA_BYTES, MODEL_INPUT_BYTES, MODEL_OUTPUT_BYTES);

    if (TEST_INPUT_BYTES != MODEL_INPUT_BYTES || TEST_OUTPUT_BYTES != MODEL_OUTPUT_BYTES) {
        printf("FAIL: test_vectors.h shape doesn't match model.h\n");
        blink_result(0);
        return 1;
    }

    int mismatches = 0;
    int first_bad = -1;
    uint8_t output[MODEL_OUTPUT_BYTES];

    uint32_t t0 = time_us_32();
    for (int i = 0; i < NUM_TEST_VECTORS; i++) {
        model_run(test_inputs[i], output);
        if (memcmp(output, test_expected[i], MODEL_OUTPUT_BYTES) != 0) {
            mismatches++;
            if (first_bad < 0) first_bad = i;
        }
    }
    uint32_t t1 = time_us_32();

    printf("Tested %d vectors in %lu us (%.1f us/inference).\n",
           NUM_TEST_VECTORS, (unsigned long)(t1 - t0),
           (t1 - t0) / (double)NUM_TEST_VECTORS);

    if (mismatches == 0) {
        printf("PASS: model_run() on real hardware matches reference.py "
               "bit-for-bit on all %d vectors.\n", NUM_TEST_VECTORS);
        blink_result(1);
    } else {
        printf("FAIL: %d/%d mismatches. First mismatch at vector %d:\n",
               mismatches, NUM_TEST_VECTORS, first_bad);
        model_run(test_inputs[first_bad], output);
        printf("  got:      ");
        for (int j = 0; j < MODEL_OUTPUT_BYTES; j++) printf("%d ", output[j]);
        printf("\n  expected: ");
        for (int j = 0; j < MODEL_OUTPUT_BYTES; j++) printf("%d ", test_expected[first_bad][j]);
        printf("\n");
        blink_result(0);
    }

    while (1) {
        tight_loop_contents();
    }
}
