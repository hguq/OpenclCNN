//
// Created by hiro on 2021/5/13.
//

#ifndef UTIL_CPP
#define UTIL_CPP

#include <bits/stdc++.h>
#include <FreeImage/FreeImage.h>

using namespace std;


void load_one_image(const string &file_path, void *buffer, size_t &w, size_t &h) {
    auto image = FreeImage_Load(FreeImage_GetFileType(file_path.c_str(), 0), file_path.c_str());
    auto temp = image;
    image = FreeImage_ConvertTo32Bits(image); // Here, 32 bits means 4 bytes for one pixel, including A
    FreeImage_Unload(temp);

    // Get width and height
    w = FreeImage_GetWidth(image);
    h = FreeImage_GetHeight(image);

    memcpy(buffer, FreeImage_GetBits(image), w * h * 4);
    FreeImage_Unload(image);
}

void load_mnist(int N, const string &file_list, const string &image_dir, unsigned char images[][784], int *labels) {
    ifstream fs(file_list);
    string file;
    // 图片需要翻转一下，零坐标点在左上角，这样才符合神经网络的输入的权重的输入模式
    unsigned char temp[1 * 28 * 28 * 4];
    for (int n = 0; n < N; n++) {
        fs >> file;
        labels[n] = file[5] - '0';
        file = image_dir + file;
        size_t image_h, image_w;
        load_one_image(file, temp, image_w, image_h);
        assert(image_w == 28 && image_h == 28);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                images[n][i * 28 + j] = temp[((27 - i) * 28 + j) * 4];
            }
        }
    }
}

void print_one_images(int images[][784], int i) {
    static ofstream fs("../images.txt");
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            fs << setw(3) << int(images[i][row * 28 + col]) << ' ';
        }
        fs << endl;
    }
}


#endif