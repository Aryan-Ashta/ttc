#include <stdio.h>
#include <string.h>
#include "model.h"
#include "test_vectors.h"

int main(void) {
    if (TEST_INPUT_BYTES != MODEL_INPUT_BYTES || TEST_OUTPUT_BYTES != MODEL_OUTPUT_BYTES) {
        printf("FAIL: test vector shape doesn't match model.h (regenerate test_vectors.h)\n");
        return 1;
    }

    int mismatches = 0;
    int first_bad = -1;
    uint8_t output[MODEL_OUTPUT_BYTES];

    for (int i = 0; i < NUM_TEST_VECTORS; i++) {
        model_run(test_inputs[i], output);
        if (memcmp(output, test_expected[i], MODEL_OUTPUT_BYTES) != 0) {
            mismatches++;
            if (first_bad < 0) first_bad = i;
        }
    }

    printf("Tested %d vectors.\n", NUM_TEST_VECTORS);
    if (mismatches == 0) {
        printf("PASS: model.c matches reference.py bit-for-bit on all %d vectors.\n", NUM_TEST_VECTORS);
        return 0;
    }

    printf("FAIL: %d/%d mismatches. First mismatch at vector %d:\n",
           mismatches, NUM_TEST_VECTORS, first_bad);
    model_run(test_inputs[first_bad], output);
    printf("  got:      ");
    for (int j = 0; j < MODEL_OUTPUT_BYTES; j++) printf("%d ", output[j]);
    printf("\n  expected: ");
    for (int j = 0; j < MODEL_OUTPUT_BYTES; j++) printf("%d ", test_expected[first_bad][j]);
    printf("\n");
    return 1;
}
