/*
 * Phase 5 hardware benchmark: naive (Phase 3) vs. optimized (Phase 5:
 * packed weight loads + hoisted per-channel unpacking + unrolled
 * K-loop) conv2d, run back-to-back on the same 500 test vectors, timed
 * with the Pico SDK's timer.
 *
 * Cortex-M0+ has no DWT cycle counter, so "cycles" here means time_us_32()
 * converted via the actual system clock frequency (clock_get_hz(clk_sys)
 *
 * Prints a comparison table over UART (GPIO0=TX, GPIO1=RX, 115200 baud)
 * and blinks the onboard LED (GPIO25, plain Pico) solid on
 * PASS / fast-blink on FAIL,.
 */
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "test_vectors.h"

extern void model_run_naive(const uint8_t *input, uint8_t *output);
extern void model_run_optimized(const uint8_t *input, uint8_t *output);

#define MODEL_OUTPUT_BYTES 36
#define MACS_PER_INFERENCE 1296  /* 4 out_ch * 6*6 out spatial * 9 taps -- conv2d only */
#define LED_PIN 25

static void blink_result(int passed) {
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    if (passed) {
        gpio_put(LED_PIN, 1);
    } else {
        while (1) {
            gpio_put(LED_PIN, 1); sleep_ms(100);
            gpio_put(LED_PIN, 0); sleep_ms(100);
        }
    }
}

typedef void (*model_fn)(const uint8_t *, uint8_t *);

static int run_and_time(const char *label, model_fn fn, uint32_t *total_us) {
    int mismatches = 0, first_bad = -1;
    uint8_t output[MODEL_OUTPUT_BYTES];

    uint32_t t0 = time_us_32();
    for (int i = 0; i < NUM_TEST_VECTORS; i++) {
        fn(test_inputs[i], output);
        if (memcmp(output, test_expected[i], MODEL_OUTPUT_BYTES) != 0) {
            mismatches++;
            if (first_bad < 0) first_bad = i;
        }
    }
    uint32_t t1 = time_us_32();
    *total_us = t1 - t0;

    double us_per_inf = (*total_us) / (double)NUM_TEST_VECTORS;
    printf("%-12s %d/%d PASS, %lu us total, %.2f us/inference\n",
           label, NUM_TEST_VECTORS - mismatches, NUM_TEST_VECTORS,
           (unsigned long)*total_us, us_per_inf);
    return mismatches;
}

int main(void) {
    stdio_init_all();
    sleep_ms(2000);

    uint32_t sys_clk_hz = clock_get_hz(clk_sys);
    printf("\n--- TTC Phase 5: naive vs. optimized conv2d on RP2040 ---\n");
    printf("system clock: %lu Hz\n", (unsigned long)sys_clk_hz);
    printf("(Cortex-M0+ has no DWT cycle counter, cycles below are\n");
    printf(" time_us_32() converted via the measured clock speed.\n");

    uint32_t us_naive, us_opt;
    int bad_naive = run_and_time("naive", model_run_naive, &us_naive);
    int bad_opt = run_and_time("optimized", model_run_optimized, &us_opt);

    if (bad_naive || bad_opt) {
        printf("\nFAIL: one or both variants mismatched the expected output.\n");
        blink_result(0);
        while (1) tight_loop_contents();
    }

    double cycles_naive = (us_naive / 1e6) * sys_clk_hz;
    double cycles_opt = (us_opt / 1e6) * sys_clk_hz;
    double cycles_per_inf_naive = cycles_naive / NUM_TEST_VECTORS;
    double cycles_per_inf_opt = cycles_opt / NUM_TEST_VECTORS;

    printf("\n--- Summary table ---\n");
    printf("%-24s %12s %12s %8s\n", "metric", "naive", "optimized", "ratio");
    printf("%-24s %12.2f %12.2f %7.2fx\n", "us / inference",
           us_naive / (double)NUM_TEST_VECTORS, us_opt / (double)NUM_TEST_VECTORS,
           (us_naive / (double)NUM_TEST_VECTORS) / (us_opt / (double)NUM_TEST_VECTORS));
    printf("%-24s %12.0f %12.0f %7.2fx\n", "cycles / inference (est)",
           cycles_per_inf_naive, cycles_per_inf_opt,
           cycles_per_inf_naive / cycles_per_inf_opt);
    printf("%-24s %12.2f %12.2f %7.2fx\n", "cycles / MAC (est)",
           cycles_per_inf_naive / MACS_PER_INFERENCE, cycles_per_inf_opt / MACS_PER_INFERENCE,
           cycles_per_inf_naive / cycles_per_inf_opt);
    printf("%-24s %12lu %12lu\n", "total us (%d vectors)",
           (unsigned long)us_naive, (unsigned long)us_opt);

    printf("\nPASS: both variants match reference.py bit-for-bit on all %d vectors.\n",
           NUM_TEST_VECTORS);
    blink_result(1);

    while (1) tight_loop_contents();
}
