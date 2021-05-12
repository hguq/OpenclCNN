#include "cnn.h"
#include <iomanip>

using namespace std;

const char *IMAGE_DIR = "C:/projects/datasets/mnist/test_images/";

unsigned char images[10000][1 * 28 * 28]; // load 10000 images, in neuron input style
int labels[10000];

void init_images() {
    ifstream fs("../image_list.txt");
    string file;

    // 图片需要翻转一下，零坐标点在左上角，这样才符合神经网络的输入的权重的输入模式
    unsigned char temp[1 * 28 * 28 * 4];
    for (int n = 0; n < 10000; n++) {
        fs >> file;
        labels[n] = file[5] - '0';
        file = IMAGE_DIR + file;
        size_t image_h, image_w;
        load_image(file, temp, image_w, image_h);
        assert(image_w == 28 && image_h == 28);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                images[n][i * 28 + j] = temp[((27 - i) * 28 + j) * 4];
            }
        }
    }
}

void print_images(int i) {
    static ofstream fs("../images.txt");
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            fs << setw(3) << int(images[i][row * 28 + col]) << ' ';
        }
        fs << endl;
    }
}

int main() {
    init_images();
    cnn cnn_instance(1, 28, 28, 10, "../kernel.cl", "../model.txt");
    for (int i = 0; i < 10; i++) {
        print_images(i);
        int result = cnn_instance.cpu_forward(images[i]);
        cout << result << endl;
    }
}
