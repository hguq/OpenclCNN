#include "cnn.h"
#include <iomanip>

using namespace std;

const char *IMAGE_DIR = "C:/projects/datasets/mnist/test_images/";
const int N_IMAGES = 10000;
unsigned char images[N_IMAGES][1 * 28 * 28]; // load N_IMAGES images, in neuron input style
int labels[N_IMAGES];

void init_images() {
    ifstream fs("../image_list.txt");
    string file;
    // 图片需要翻转一下，零坐标点在左上角，这样才符合神经网络的输入的权重的输入模式
    unsigned char temp[1 * 28 * 28 * 4];
    for (int n = 0; n < N_IMAGES; n++) {
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
    print_images(0);
    cnn cnn_instance(1, 28, 28, 10, "../kernel.cl", "../model.txt");

    int correct = 0;
    for (int i = 0; i < 100; i++) {
        int result = cnn_instance.cpu_forward(images[i]);
        if (result == labels[i])++correct;
        cout<<result<<endl;
    }
    cout<<(correct);
}
