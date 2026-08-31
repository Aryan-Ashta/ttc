#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "test_vectors.h"

extern void model_run_naive(const uint8_t *input, uint8_t *output);
extern void model_run_optimized(const uint8_t *input, uint8_t *output);

#define MODEL_OUTPUT_BYTES 36

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int run_and_check(const char *label, void (*fn)(const uint8_t *, uint8_t *), double *seconds_out) {
    int mismatches = 0, first_bad = -1;
    uint8_t output[MODEL_OUTPUT_BYTES];

    double t0 = now_seconds();
    for (int i = 0; i < NUM_TEST_VECTORS; i++) {
        fn(test_inputs[i], output);
        if (memcmp(output, test_expected[i], MODEL_OUTPUT_BYTES) != 0) {
            mismatches++;
            if (first_bad < 0) first_bad = i;
        }
    }
    double t1 = now_seconds();
    *seconds_out = t1 - t0;

    printf("%-20s %d/%d vectors, %s, %.3f ms total (%.2f us/inference) "
           "[host timing -- NOT representative of on-device cycles]\n",
           label, NUM_TEST_VECTORS - mismatches, NUM_TEST_VECTORS,
           mismatches == 0 ? "PASS" : "FAIL",
           *seconds_out * 1000.0, *seconds_out * 1e6 / NUM_TEST_VECTORS);
    return mismatches;
}

int main(void) {
    double t_naive, t_opt;
    int bad_naive = run_and_check("naive", model_run_naive, &t_naive);
    int bad_opt = run_and_check("optimized", model_run_optimized, &t_opt);

    if (bad_naive || bad_opt) {
        printf("FAIL: one or both variants produced mismatches vs. reference.py.\n");
        return 1;
    }
    printf("\nBoth variants bit-exact against reference.py on all %d vectors.\n", NUM_TEST_VECTORS);
    printf("Host wall-clock ratio (naive/optimized): %.2fx "
           "-- again, host timing only, NOT the real answer; see the Pico firmware.\n",
           t_naive / t_opt);
    return 0;
}
