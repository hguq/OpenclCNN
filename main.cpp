#include "cnn.cpp"
#include "timer.cpp"
#include "util.cpp"

using namespace std;

const char IMAGE_DIR[] = "../test_images/";
const char IMAGE_LIST_FILE[] = "../image_list.txt";
const char KERNEL_FILE[] = "../kernel.cl";
const char MODEL_FILE[] = "../model.txt";
const int N_IMAGES = 10000;
const int N_TESTS = 10000;
uint8_t images[N_IMAGES][1 * 28 * 28];
int labels[N_IMAGES];


int main() {
    load_mnist(N_IMAGES, IMAGE_LIST_FILE, IMAGE_DIR, images, labels);
    cnn cnn_instance(1, 28, 28, 10,
                     KERNEL_FILE, MODEL_FILE);

    int correct = 0;
    for (int i = 0; i < N_TESTS; i++)if (cnn_instance.opencl_forward(images[i]) == labels[i])++correct;

    cout << "OPENCL CORRECT: " << correct << '/' << N_TESTS << endl;
    cnn_instance.report_opencl_time();

    correct = 0;
    for (int i = 0; i < N_TESTS; i++)if (cnn_instance.cpu_forward(images[i]) == labels[i])++correct;

    cout << "CPU CORRECT: " << correct << '/' << N_TESTS << endl;

    cnn_instance.report_cpu_time();

    return 0;
}
