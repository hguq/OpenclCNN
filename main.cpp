#include "cnn.cpp"
#include "timer.cpp"
#include "util.cpp"

using namespace std;

const int N_IMAGES = 10000;
const int N_TESTS = 10000;
unsigned char images[N_IMAGES][1 * 28 * 28];
int labels[N_IMAGES];


int main() {
    load_mnist(N_IMAGES, "../image_list.txt", "../test_images/", images, labels);
    cnn cnn_instance(1, 28, 28, 10, "../kernel.cl", "../model.txt");

    int correct = 0;
    start_timer();
    for (int i = 0; i < N_TESTS; i++)if (cnn_instance.opencl_forward(images[i]) == labels[i])++correct;
    float time = end_timer();

    cout << "OPENCL TIME: " << time << " sec." << endl;
    cout << "OPENCL CORRECT: " << correct << '/' << N_TESTS << endl;

    correct=0;
    start_timer();
    for (int i = 0; i < N_TESTS; i++)if (cnn_instance.cpu_forward(images[i]) == labels[i])++correct;
    time = end_timer();

    cout << "CPU TIME: " << time << " sec." << endl;
    cout << "CPU CORRECT: " << correct << '/' << N_TESTS << endl;
}
