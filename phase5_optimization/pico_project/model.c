#include "model.h"
#include <math.h>
#include <string.h>

static uint8_t arena[MODEL_ARENA_BYTES];

/* Byte offsets into arena[], from Phase 2's memory planner. */
#define OFFSET_input 144
#define OFFSET_conv_out 0
#define OFFSET_relu_out 0
#define OFFSET_pool_out 144

/* Weights & biases, baked in as flash constants. */
static const int8_t conv1_weight[4][1][3][3] = {{{{7, -66, 22}, {-120, -93, -66}, {81, 75, -57}}}, {{{-5, 82, 127}, {51, 17, 86}, {-76, 24, -99}}}, {{{-89, -66, 58}, {52, -76, 39}, {70, -16, 5}}}, {{{30, 80, 123}, {-99, -47, 50}, {106, 112, 113}}}};
static const int32_t conv1_bias[4] = {814, -3555, 376, -2558};

static uint8_t requantize(int32_t acc, double multiplier, int32_t zero_point) {
    double scaled = (double)acc * multiplier;
    long q = (long)nearbyint(scaled) + zero_point;
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (uint8_t)q;
}

static void op0_conv2d(void) {
    const uint8_t *in = &arena[OFFSET_input];
    uint8_t *out = &arena[OFFSET_conv_out];
    for (int oc = 0; oc < 4; oc++) {
        for (int oy = 0; oy < 6; oy++) {
            for (int ox = 0; ox < 6; ox++) {
                int32_t acc = conv1_bias[oc];
                for (int ic = 0; ic < 1; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int32_t x = (int32_t)in[ic*8*8 + (oy+ky)*8 + (ox+kx)]
                                        - 127;
                            int32_t w = conv1_weight[oc][ic][ky][kx];
                            acc += x * w;
                        }
                    }
                }
                out[oc*6*6 + oy*6 + ox] =
                    requantize(acc, 0.001768886219063262, 136);
            }
        }
    }
}

static void op1_relu(void) {
    const uint8_t *in = &arena[OFFSET_conv_out];
    uint8_t *out = &arena[OFFSET_relu_out];
    for (int i = 0; i < 144; i++) {
        uint8_t v = in[i];
        out[i] = v < 136 ? 136 : v;
    }
}

static void op2_maxpool2d(void) {
    const uint8_t *in = &arena[OFFSET_relu_out];
    uint8_t *out = &arena[OFFSET_pool_out];
    for (int c = 0; c < 4; c++) {
        for (int oy = 0; oy < 3; oy++) {
            for (int ox = 0; ox < 3; ox++) {
                uint8_t m = in[c*6*6 + (oy*2)*6 + (ox*2)];
                for (int py = 0; py < 2; py++) {
                    for (int px = 0; px < 2; px++) {
                        uint8_t v = in[c*6*6 + (oy*2+py)*6 + (ox*2+px)];
                        if (v > m) m = v;
                    }
                }
                out[c*3*3 + oy*3 + ox] = m;
            }
        }
    }
}

void model_run_naive(const uint8_t *input, uint8_t *output) {
    memcpy(&arena[OFFSET_input], input, MODEL_INPUT_BYTES);
    op0_conv2d();
    op1_relu();
    op2_maxpool2d();
    memcpy(output, &arena[OFFSET_pool_out], MODEL_OUTPUT_BYTES);
}
