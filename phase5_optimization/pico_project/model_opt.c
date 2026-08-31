#include "model_opt.h"
#include <math.h>
#include <string.h>

static uint8_t arena[MODEL_ARENA_BYTES];

/* Byte offsets into arena[], from Phase 2's memory planner. */
#define OFFSET_input 144
#define OFFSET_conv_out 0
#define OFFSET_relu_out 0
#define OFFSET_pool_out 144

/* Weights & biases, baked in as flash constants. */
static const int32_t conv1_bias[4] = {814, -3555, 376, -2558};
/* Phase 5: same weight bytes, padded to a multiple of 4 per channel and
 * 4-byte aligned, so they can be read as packed uint32_t words. */
static const uint8_t conv1_weight_packed[4][12] __attribute__((aligned(4))) = {{7, 190, 22, 136, 163, 190, 81, 75, 199, 0, 0, 0}, {251, 82, 127, 51, 17, 86, 180, 24, 157, 0, 0, 0}, {167, 190, 58, 52, 180, 39, 70, 240, 5, 0, 0, 0}, {30, 80, 123, 157, 209, 50, 106, 112, 113, 0, 0, 0}};

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
    const uint32_t *wpacked = (const uint32_t *)conv1_weight_packed;
    for (int oc = 0; oc < 4; oc++) {
        /* unpack 9 weight bytes (3 packed uint32_t reads) once per channel,
           manual sign-extension via shift trick (M0+ has no SXTB16) */
        int32_t w[9];
        {
            uint32_t packed = wpacked[oc*3 + 0];
            w[0] = ((int32_t)(packed << 24)) >> 24;
            w[1] = ((int32_t)(packed << 16)) >> 24;
            w[2] = ((int32_t)(packed << 8)) >> 24;
            w[3] = ((int32_t)(packed << 0)) >> 24;
        }
        {
            uint32_t packed = wpacked[oc*3 + 1];
            w[4] = ((int32_t)(packed << 24)) >> 24;
            w[5] = ((int32_t)(packed << 16)) >> 24;
            w[6] = ((int32_t)(packed << 8)) >> 24;
            w[7] = ((int32_t)(packed << 0)) >> 24;
        }
        {
            uint32_t packed = wpacked[oc*3 + 2];
            w[8] = ((int32_t)(packed << 24)) >> 24;
        }
        for (int oy = 0; oy < 6; oy++) {
            for (int ox = 0; ox < 6; ox++) {
                int32_t acc = conv1_bias[oc];
                /* fully unrolled K=9 MAC -- no loop counter, no branch */
                acc += ((int32_t)in[0*8*8 + (oy+0)*8 + (ox+0)] - 127) * w[0];
                acc += ((int32_t)in[0*8*8 + (oy+0)*8 + (ox+1)] - 127) * w[1];
                acc += ((int32_t)in[0*8*8 + (oy+0)*8 + (ox+2)] - 127) * w[2];
                acc += ((int32_t)in[0*8*8 + (oy+1)*8 + (ox+0)] - 127) * w[3];
                acc += ((int32_t)in[0*8*8 + (oy+1)*8 + (ox+1)] - 127) * w[4];
                acc += ((int32_t)in[0*8*8 + (oy+1)*8 + (ox+2)] - 127) * w[5];
                acc += ((int32_t)in[0*8*8 + (oy+2)*8 + (ox+0)] - 127) * w[6];
                acc += ((int32_t)in[0*8*8 + (oy+2)*8 + (ox+1)] - 127) * w[7];
                acc += ((int32_t)in[0*8*8 + (oy+2)*8 + (ox+2)] - 127) * w[8];
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

void model_run_optimized(const uint8_t *input, uint8_t *output) {
    memcpy(&arena[OFFSET_input], input, MODEL_INPUT_BYTES);
    op0_conv2d();
    op1_relu();
    op2_maxpool2d();
    memcpy(output, &arena[OFFSET_pool_out], MODEL_OUTPUT_BYTES);
}
