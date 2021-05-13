//
// Created by hiro on 2021/5/10.
//

#ifndef CNN_H
#define CNN_H

#include <CL/opencl.h>
#include <FreeImage/FreeImage.h>
#include "func.h"

using namespace std;

int ret;
#define check assert(ret==0);

template<class T>
void assert_array(T *A, T *B, size_t N) {
    for (int n = 0; n < N; n++) {
        if (A[n] != B[n]) {
            cout<<"Wrong"<<endl;
        }
    }

}

class layer {
public:
    // Opencl related variables ( pointers )
    cl_command_queue command_queue = nullptr;
    cl_kernel kernel = nullptr;

    vector<cl_mem> allocated;

    // Set a pointer for cpu_out.
    // Set a mem buffer for opencl_out.
    // Both will be allocated in constructor.
    void *cpu_out = nullptr;
    cl_mem opencl_out = nullptr;

    // Set kernel work dimension.
    // Always use 3-dimension.
    size_t *global_work_size = nullptr;
    size_t *local_work_size = nullptr;

    // Initialize the layer
    // pass program and let subsidiary classes create kernels by themselves.
    explicit layer(cl_command_queue command_queue_) : command_queue(command_queue_) {}

    // Pure virtual function that do cpu forward propagation.
    virtual void *cpu_forward(void *input) = 0;

    // Pure virtual function that do opencl_forward propagation.
    // Calculate result and put result in "opencl_out" buffer, and return it.
    virtual cl_mem opencl_forward(void *cpu_in, cl_mem opencl_in) = 0;

    virtual ~layer() {
        for (auto ptr:allocated) {
            clReleaseMemObject(ptr);
        }
        clReleaseKernel(kernel);
    }
};

class conv_layer : public layer {
public:
    size_t CI, CO, H, W;

    cl_mem opencl_weight = nullptr;
    signed char *cpu_weight = nullptr;

    conv_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t CI_, size_t CO_, size_t H_, size_t W_, signed char *weight_ptr) :
            layer(command_queue_),
            CI(CI_), CO(CO_), H(H_), W(W_) {
        // Create kernel
        kernel = clCreateKernel(program_, "conv", &ret);
        check


        // Save cpu opencl_weight and allocate space for cpu output
        cpu_weight = weight_ptr;
        cpu_out = new int32_t[CO * H * W];

        // Create opencl_weight and result buffer;
        opencl_weight = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       CO * CI * 3 * 3 * sizeof(signed char), (void *) weight_ptr, &ret);
        check
        opencl_out = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                    CO * H * W * sizeof(int32_t), nullptr, &ret);
        check

        // Record allocated cl mem
        allocated.push_back(opencl_weight);
        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{H, W, CO};
        local_work_size = nullptr;

    }

    void *cpu_forward(void *input) override {
        // Call cpu version conv function here
        cpu_conv(CI, CO, H, W,
                 (const signed char *) cpu_weight, (const unsigned char *) input, (int32_t *) cpu_out);
        return cpu_out;
    }

    //  Set argument and execute kernel.
    cl_mem opencl_forward(void *cpu_in, cl_mem opencl_in) override {
        // input is unsigned char
        // unsigned char *+ signed char -> int32_t
        // Set kernel argument
        // Because the arguments never change, so specify them in constructor.
        cpu_forward(cpu_in);// The result is stored in local cpu_out
        ret = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &CI);
        check
        ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &CO);
        check
        ret = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &H);
        check
        ret = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &W);
        check
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &opencl_weight);
        check
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &opencl_in);
        check
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &opencl_out);
        check
        // Execute kernel
        ret = clEnqueueNDRangeKernel(command_queue,
                                     kernel,
                                     3, // Dimension
                                     nullptr, // Global offset
                                     global_work_size, // Global work size
                                     local_work_size, // Local work size
                                     0, // Number of events in wait list
                                     nullptr, // Wait list
                                     nullptr // Bounding event
        );
        check

        auto opencl_out_copy = new int32_t[CO * H * W];
        clEnqueueReadBuffer(command_queue, opencl_out, CL_TRUE, 0, CO * H * W * sizeof(int32_t),
                            opencl_out_copy, 0, nullptr, nullptr);
        assert_array((int32_t *) cpu_out, (int32_t *) opencl_out_copy, CO * H * W);
        return opencl_out;
    }


    ~conv_layer() override {
        delete[] cpu_weight;
        delete[] (int32_t *) cpu_out;
    }
};

class fc_layer : public layer {
public:
    size_t CI, CO;

    cl_mem opencl_weight;
    signed char *cpu_weight;

    fc_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
             size_t CI_, size_t CO_, signed char *weight_ptr) :
            layer(command_queue_), CI(CI_), CO(CO_) {
        // Create kernel
        kernel = clCreateKernel(program_, "fc", &ret);
        check

        // Save cpu weight and allocate space for cpu output
        cpu_weight = weight_ptr;
        cpu_out = new int32_t[CO];

        // Create opencl_weight and result buffer;
        opencl_weight = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       CI * CO * sizeof(signed char), (void *) weight_ptr, &ret);
        check
        opencl_out = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                    CO * sizeof(int32_t), nullptr, &ret);
        check

        allocated.push_back(opencl_weight);
        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{CO, 1, 1};
        local_work_size = nullptr;
    }

    cl_mem opencl_forward(void *cpu_in, cl_mem opencl_in) override {
        cpu_forward(cpu_in);
        // Set arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &CI);
        check
        ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &CO);
        check
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &opencl_weight);
        check
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &opencl_in);
        check
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &opencl_out);
        check
        // input: unsigned char
        // unsigned char *+ signed char -> int32_t
        ret = clEnqueueNDRangeKernel(command_queue,
                                     kernel,
                                     3, // Dimension
                                     nullptr, // Global offset
                                     global_work_size, // Global work size
                                     local_work_size, // Local work size
                                     0, // Number of events in wait list
                                     nullptr, // Wait list
                                     nullptr // Bounding event
        );
        check

        auto opencl_out_copy = new int32_t[CO];
        clEnqueueReadBuffer(command_queue, opencl_out, CL_TRUE, 0, CO * sizeof(int32_t),
                            opencl_out_copy, 0, nullptr, nullptr);
        assert_array((int32_t *) cpu_out, (int32_t *) opencl_out_copy, CO);

        return opencl_out;
    }

    void *cpu_forward(void *input) override {
        cpu_fc(CI, CO, (const signed char *) cpu_weight, (const unsigned char *) input, (int32_t *) cpu_out);
        return cpu_out;
    }


    ~fc_layer() override {
        delete[] cpu_weight;
        delete[] (int32_t *) cpu_out;
    }
};

class quan_layer : public layer {
public:
    size_t C, H, W; // If is fc, H == W == 1

    cl_mem opencl_bias, opencl_shift;

    int32_t *cpu_bias = nullptr;
    unsigned char *cpu_shift = nullptr;

    quan_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t C_, size_t H_, size_t W_, int32_t *bias_ptr, unsigned char *shift_ptr) :
            layer(command_queue_), C(C_), H(H_), W(W_) {
        // Create kernel
        kernel = clCreateKernel(program_, "quan", &ret);
        check

        // Save cpu bias and shift
        cpu_bias = bias_ptr;
        cpu_shift = shift_ptr;
        cpu_out = new signed char[C * H * W];

        // Create opencl_weight and result buffer;
        opencl_bias = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, C * sizeof(int32_t),
                                     (void *) bias_ptr, &ret);
        check
        opencl_shift = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, C * sizeof(unsigned char),
                                      (void *) shift_ptr, &ret);
        check
        opencl_out = clCreateBuffer(context_, CL_MEM_READ_WRITE, C * H * W * sizeof(signed char),
                                    nullptr, &ret);
        check

        allocated.push_back(opencl_bias);
        allocated.push_back(opencl_shift);
        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{H, W, C};
        local_work_size = nullptr;
    }

    cl_mem opencl_forward(void *cpu_in, cl_mem opencl_in) override {
        cpu_forward(cpu_in);
        // Set kernel arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &C);
        check
        ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &H);
        check
        ret = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &W);
        check
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &opencl_bias);
        check
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &opencl_shift);
        check
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &opencl_in);
        check
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &opencl_out);
        check
        // input is int32_t
        // (int32_t - int32_t) >> unsigned char -> signed char
        ret = clEnqueueNDRangeKernel(command_queue,
                                     kernel,
                                     3, // Dimension
                                     nullptr, // Global offset
                                     global_work_size, // Global work size
                                     local_work_size, // Local work size
                                     0, // Number of events in wait list
                                     nullptr, // Wait list
                                     nullptr // Bounding event
        );
        check
        auto opencl_out_copy = new signed char[C * H * W];
        clEnqueueReadBuffer(command_queue, opencl_out, CL_TRUE, 0, C * H * W * sizeof(signed char),
                            opencl_out_copy, 0, nullptr, nullptr);
        assert_array((signed char *) cpu_out, (signed char *) opencl_out_copy, C * H * W);
        return opencl_out;
    }

    void *cpu_forward(void *input) override {
        cpu_quan(C, H, W,
                 (const int32_t *) cpu_bias,
                 (const unsigned char *) cpu_shift,
                 (const int32_t *) input,
                 (signed char *) cpu_out);
        return cpu_out;
    }

    ~quan_layer() override {
        delete[] cpu_bias;
        delete[] cpu_shift;
        delete[] (signed char *) cpu_out;
    }


};

class pool_layer : public layer {
public:
    size_t C, H, W;
    size_t HO, WO;

    pool_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t C_, size_t H_, size_t W_) :
            layer(command_queue_), C(C_), H(H_), W(W_) {
        // Calculate opencl_out height and width
        HO = H >> 1u;
        WO = W >> 1u;
        // Create kernel
        kernel = clCreateKernel(program_, "pool", &ret);
        check


        // allocate space for cpu out
        cpu_out = new signed char[C * HO * WO];

        // Create opencl_weight and result buffer;
        opencl_out = clCreateBuffer(context_, CL_MEM_READ_WRITE, C * HO * WO * sizeof(signed char), nullptr, nullptr);

        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{HO, WO, C};
        local_work_size = nullptr;
    }

    cl_mem opencl_forward(void *cpu_in, cl_mem opencl_in) override {
        cpu_forward(cpu_in);
        // Set kernel arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &C);
        check
        ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &H);
        check
        ret = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &W);
        check
        ret = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &HO);
        check
        ret = clSetKernelArg(kernel, 4, sizeof(cl_ulong), &WO);
        check
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &opencl_in);
        check
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &opencl_out);
        check
        // input is unsigned char
        // unsigned char -> unsigned char
        ret = clEnqueueNDRangeKernel(command_queue,
                                     kernel,
                                     3, // Dimension
                                     nullptr, // Global offset
                                     global_work_size, // Global work size
                                     local_work_size, // Local work size
                                     0, // Number of events in wait list
                                     nullptr, // Wait list
                                     nullptr // Bounding event
        );
        check
        auto opencl_out_copy = new unsigned char[C * H * W];
        clEnqueueReadBuffer(command_queue, opencl_out, CL_TRUE, 0, C * H * W * sizeof(unsigned char),
                            opencl_out_copy, 0, nullptr, nullptr);
        assert_array((unsigned char *) cpu_out, (unsigned char *) opencl_out_copy, C * H * W);
        return opencl_out;
    }

    void *cpu_forward(void *input) override {
        cpu_pool(C, H, W, HO, WO, (unsigned char *) input, (unsigned char *) cpu_out);
        return cpu_out;
    }

    ~pool_layer() override {
        delete[] (signed char *) cpu_out;
    }

};

class relu_layer : public layer {
public:
    size_t C, H, W;

    relu_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t C_, size_t H_, size_t W_) :
            layer(command_queue_), C(C_), H(H_), W(W_) {
        // Create kernel
        kernel = clCreateKernel(program_, "relu", &ret);
        check

        // Allocate space for cpu output
        cpu_out = new unsigned char[C * H * W];

        // Create opencl_weight and result buffer;
        opencl_out = clCreateBuffer(context_, CL_MEM_READ_WRITE, C * H * W * sizeof(signed char), nullptr, nullptr);

        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{C, H, W};
        local_work_size = nullptr;
    }

    cl_mem opencl_forward(void *cpu_in, cl_mem opencl_in) override {
        cpu_forward(cpu_in);
        // Set kernel arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &C);
        check
        ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &H);
        check
        ret = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &W);
        check
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &opencl_in);
        check
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &opencl_out);
        check
        // input is signed char
        // signed char -> unsigned char
        ret = clEnqueueNDRangeKernel(command_queue,
                                     kernel,
                                     3, // Dimension
                                     nullptr, // Global offset
                                     global_work_size, // Global work size
                                     local_work_size, // Local work size
                                     0, // Number of events in wait list
                                     nullptr, // Wait list
                                     nullptr // Bounding event
        );
        check
        auto opencl_out_copy = new unsigned char[C * H * W];
        clEnqueueReadBuffer(command_queue, opencl_out, CL_TRUE, 0, C * H * W * sizeof(unsigned char),
                            opencl_out_copy, 0, nullptr, nullptr);
        assert_array((unsigned char *) cpu_out, (unsigned char *) opencl_out_copy, C * H * W);
        return opencl_out;
    }

    void *cpu_forward(void *input) override {
        cpu_relu(C, H, W, (signed char *) input, (unsigned char *) cpu_out);
        return cpu_out;
    }

    ~relu_layer() override {
        delete[] (signed char *) cpu_out;
    }


};

class cnn {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;

    cl_mem opencl_in = nullptr;
    cl_mem opencl_out = nullptr;
    signed char *cpu_out = nullptr;

    vector<layer *> layers;

    size_t IMAGE_C, IMAGE_H, IMAGE_W, FEATURE;
public:
    void parse_model_file(const string &model_file) {
        ifstream fs(model_file);
        string s;
        int tmp;
        while (fs >> s) {
            if (s == "CONV") {
                int CO, CI, H, W;
                fs >> s;
                assert(s == "CO");
                fs >> CO >> s;
                assert(s == "CI");
                fs >> CI >> s;
                assert(s == "H");
                fs >> H >> s;
                assert(s == "W");
                fs >> W;
                auto weight_ptr = new signed char[CO * CI * 3 * 3];
                for (int co = 0; co < CO; co++) {
                    for (int ci = 0; ci < CI; ci++) {
                        for (int h = 0; h < 3; h++) {
                            for (int w = 0; w < 3; w++) {
                                fs >> tmp;
                                weight_ptr[co * CI * 3 * 3 + ci * 3 * 3 + h * 3 + w] = tmp;
                            }
                        }
                    }
                }
                layers.emplace_back(new conv_layer(context, command_queue, program, CI, CO, H, W, weight_ptr));
            } else if (s == "FC") {
                int CI, CO;
                fs >> s;
                assert(s == "CI");
                fs >> CI >> s;
                assert(s == "CO");
                fs >> CO;
                auto weight_ptr = new signed char[CI * CO];
                for (int ci = 0; ci < CI; ci++) {
                    for (int co = 0; co < CO; co++) {
                        fs >> tmp;
                        weight_ptr[ci * CO + co] = tmp;
                    }
                }
                layers.emplace_back(new fc_layer(context, command_queue, program, CI, CO, weight_ptr));
            } else if (s == "RELU") {
                int C, H, W;
                fs >> s;
                assert(s == "C");
                fs >> C >> s;
                assert(s == "H");
                fs >> H >> s;
                assert(s == "W");
                fs >> W;
                layers.emplace_back(new relu_layer(context, command_queue, program, C, H, W));
            } else if (s == "POOL") {
                int C, H, W;
                fs >> s;
                assert(s == "C");
                fs >> C >> s;
                assert(s == "H");
                fs >> H >> s;
                assert(s == "W");
                fs >> W;
                layers.emplace_back(new pool_layer(context, command_queue, program, C, H, W));
            } else if (s == "QUAN") {
                int C, H, W;
                fs >> s;
                assert(s == "C");
                fs >> C >> s;
                assert(s == "H");
                fs >> H >> s;
                assert(s == "W");
                fs >> W;
                auto bias_ptr = new int32_t[C];
                auto shift_ptr = new unsigned char[C];
                fs >> s;
                assert(s == "BIAS");
                for (int c = 0; c < C; c++) {
                    fs >> tmp;
                    bias_ptr[c] = tmp;
                }
                fs >> s;
                assert(s == "SHIFT");
                for (int c = 0; c < C; c++) {
                    fs >> tmp;
                    shift_ptr[c] = tmp;
                }
                layers.emplace_back(new quan_layer(context, command_queue, program, C, H, W, bias_ptr, shift_ptr));
            } else {
                cout << "No such layer: " << s << endl;
                exit(1);
            }
        }
    }

    cnn(size_t IMAGE_C_, size_t IMAGE_H_, size_t IMAGE_W_, size_t FEATURE_,
        const string &kernel_file, const string &model_file) :
            IMAGE_C(IMAGE_C_), IMAGE_H(IMAGE_H_), IMAGE_W(IMAGE_W_), FEATURE(FEATURE_) {
        opencl_init(kernel_file);
        parse_model_file(model_file);
        cpu_out = new signed char[FEATURE];
    }

    static string read_file(const string &file_path) {
        ifstream ifs(file_path);
        stringstream ss;
        ss << ifs.rdbuf();
        return ss.str();
    }

    void opencl_init(const string &kernel_file) {
        ret = clGetPlatformIDs(1, &platform, nullptr);
        check
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
        check

        // Create context & queue.
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
        check
        command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
        check

        // Read program source file & create kernel
        string src = read_file(kernel_file);
        program = clCreateProgramWithSource(context, 1, (const char **) &src, nullptr, &ret);
        check
        ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (ret) {
            size_t log_size;
            ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            auto build_log = new char[log_size];
            build_log[log_size] = '\n';
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, nullptr);
            cout << build_log;
            exit(1);
        }
        opencl_in = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   IMAGE_C * IMAGE_H * IMAGE_W * sizeof(unsigned char), nullptr, &ret);
        check
        opencl_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    FEATURE * sizeof(signed char), nullptr, &ret);
        check
    }


    void opencl_release() {
        // Release.
        // Kernels will be released in the deconstruct function of layers
        clReleaseMemObject(opencl_in);
        clReleaseMemObject(opencl_out);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }


    ~cnn() {
        opencl_release();
        for (auto layer:layers) {
            delete layer;
        }
    }

    template<class T>
    static size_t argmax(T *arr, int N) {
        T max_rc = *arr;
        size_t rc = 0;
        size_t ptr = 1;
        while (ptr < N) {
            if (arr[ptr] > max_rc) {
                max_rc = arr[ptr];
                rc = ptr;
            }
            ++ptr;
        }
        return rc;
    }

    size_t opencl_forward(unsigned char *image) {
        ret = clEnqueueWriteBuffer(command_queue, opencl_in, CL_TRUE, 0,
                                   IMAGE_C * IMAGE_H * IMAGE_W * sizeof(unsigned char),
                                   image, 0, nullptr, nullptr);
        check
        cl_mem opencl_cur = opencl_in;
        void *cpu_cur = image;
        for (auto layer:layers) {
            opencl_cur = layer->opencl_forward(cpu_cur, opencl_cur);
            cpu_cur = layer->cpu_out;
        }

        ret = clEnqueueReadBuffer(command_queue, opencl_out, CL_TRUE, 0, FEATURE * sizeof(signed char),
                                  cpu_out, 0, nullptr, nullptr);
        check
        return argmax(cpu_out, FEATURE);
    }

    size_t cpu_forward(unsigned char *image) {
        void *cur = image;
        //for (auto layer:layers)
        //    cur = layer->cpu_forward(cur);
        for (auto &layer : layers)
            cur = layer->cpu_forward(cur);
        memcpy(cpu_out, cur, FEATURE * sizeof(signed char)); // quantized output
        return argmax(cpu_out, FEATURE);
    }
};

void load_image(const string &file_path, void *buffer, size_t &w, size_t &h) {
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


#endif //CNN_H