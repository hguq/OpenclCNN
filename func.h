//
// Created by hiro on 2021/5/12.
//

#ifndef OPENCL_CNN_CONV_FUNC_H
#define OPENCL_CNN_CONV_FUNC_H

#include <bits/stdc++.h>

using namespace std;


void conv(size_t CI, size_t CO, size_t H, size_t W,
          const signed char *weight,
          const unsigned char *image,
          int32_t *dst) {
    for (int co = 0; co < CO; co++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int32_t acc = 0;
                for (int dw = -1; dw <= 1; dw++) {
                    for (int dh = -1; dh <= 1; dh++) {
                        int hh = h + dh, ww = w + dw;
                        int hhh = dh + 1, www = dw + 1;
                        if (ww >= 0 && ww < W && hh >= 0 && hh < H) {
                            for (int ci = 0; ci < CI; ci++) {
                                acc += weight[co * CI * 3 * 3 + ci * 3 * 3 + hhh * 3 + www] *
                                       image[ci * H * W + hh * W + ww];
                            }
                        }
                    }
                }
                dst[co * H * W + h * W + w] = acc;
            }
        }
    }
}

void fc(size_t CI, size_t CO,
        const signed char *weight,
        const unsigned char *feature,
        int32_t *dst) {
    for (int co = 0; co < CO; co++) {
        int acc = 0;
        for (int ci = 0; ci < CI; ci++) {
            acc += feature[ci] * weight[ci * CO + co];
        }
        dst[co] = acc;
    }
}

void quan(size_t C, size_t H, size_t W,
          const int32_t *bias,
          const unsigned char *shift,
          const int32_t *feature,
          signed char *dst) {

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int pos = c * H * W + h * W + w;
                int32_t res = (feature[pos] - bias[c]) >> shift[c];
                dst[pos] = res;
            }
        }
    }
}

void pool(size_t C, size_t H, size_t W, size_t HO, size_t WO,
          unsigned char *feature,
          unsigned char *dst) {
    for (int c = 0; c < C; c++) {
        for (int ho = 0; ho < HO; ho++) {
            for (int wo = 0; wo < WO; wo++) {
                unsigned char result = 0;
                for (int dh = 0; dh <= 1; dh++) {
                    for (int dw = 0; dw <= 1; dw++) {
                        int h = ho * 2 + dh, w = wo * 2 + dw;
                        if (h >= 0 && h < H && w > 0 && w < W) {
                            result = max(result, feature[c * H * W + h * W + w]);
                        }
                    }
                }
                dst[c * HO * WO + ho * WO + wo] = result;
            }
        }
    }
}


void relu(size_t C, size_t H, size_t W,
          signed char *feature,
          unsigned char *dst) {
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int pos = c * H * W + h * W + w;
                dst[pos] = max((signed char) 0, feature[pos]);
            }
        }
    }
}

#endif //OPENCL_CNN_CONV_FUNC_H
