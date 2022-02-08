//
// Created by hiro on 2021/5/10.
//

#ifndef OPENCL_CNN_CONV_CNN_CPP
#define OPENCL_CNN_CONV_CNN_CPP

#define TEST_PART_TIME// test every single part time

#include <CL/opencl.h>
#include <FreeImage/FreeImage.h>
#include "func.cpp"
#include "timer.cpp"

using namespace std;

int ret;
#define check assert(ret==0);

class layer {
public:
    // Time for forwarding propagation.
    double cpu_time = 0, opencl_time = 0;
    cl_event exec_event = nullptr;

    // Opencl related variables ( pointers )
    cl_command_queue command_queue = nullptr;
    cl_kernel kernel = nullptr;

    // Allocated opencl buffers. Will be released in destructor.
    vector<cl_mem> allocated;

    // Set a pointer for out_buff.
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
    explicit layer(cl_command_queue command_queue_) :
            command_queue(command_queue_), cpu_time(0), opencl_time(0) {}

    // Pure virtual function that do cpu forward propagation.
    virtual void *cpu_forward(void *input) = 0;

    // Pure virtual function that set opencl kernel args
    virtual void opencl_set_args(cl_mem opencl_in) = 0;

    // Pure virtual function that do opencl_forward propagation.
    // Calculate result and put result in "opencl_out" buffer, and return it.
    cl_mem opencl_forward(cl_mem opencl_in) {
        opencl_set_args(opencl_in);
        // Execute kernel
        ret = clEnqueueNDRangeKernel(command_queue,
                                     kernel,
                                     3, // Dimension
                                     nullptr, // Global offset
                                     global_work_size, // Global work size
                                     local_work_size, // Local work size
                                     0, // Number of events in wait list
                                     nullptr, // Wait list
                                     &exec_event // Bounding event
        );
#ifdef TEST_PART_TIME
        accumulate_opencl_time();
#endif
        check
        // Get executing time;
        return opencl_out;
    };

    virtual ~layer() {
        for (auto ptr:allocated) {
            clReleaseMemObject(ptr);
        }
        clReleaseKernel(kernel);
    }

    virtual string type() = 0;

    virtual void report_cpu_time() { cout << type() << ": " << cpu_time << endl; }

    virtual void report_opencl_time() { cout << type() << ": " << opencl_time << endl; }

    void accumulate_opencl_time() {
        clFinish(command_queue);
        cl_ulong op, ed;
        ret = clGetEventProfilingInfo(exec_event, CL_PROFILING_COMMAND_START, sizeof(op), &op, nullptr);
        check
        ret = clGetEventProfilingInfo(exec_event, CL_PROFILING_COMMAND_END, sizeof(ed), &ed, nullptr);
        check
        double kernel_time = double(ed - op) / 1e9;
        opencl_time += kernel_time;
    }
};

class conv_layer : public layer {
public:
    size_t CI, CO, H, W;
    cl_mem opencl_weight = nullptr;
    int8_t *cpu_weight = nullptr;

    string type() override { return "conv"; }

    conv_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t CI_, size_t CO_, size_t H_, size_t W_, int8_t *weight_ptr) :
            layer(command_queue_),
            CI(CI_), CO(CO_), H(H_), W(W_) {
        // Create kernel
        kernel = clCreateKernel(program_, "conv", &ret);
        check
        // Save cpu opencl_weight and allocate space for cpu output
        cpu_weight = weight_ptr;
        cpu_out = new int32_t[CO * H * W];
        // Create opencl_weight and result buffer;
        opencl_weight = clCreateBuffer(context_,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, // Token
                                       CO * CI * 3 * 3 * sizeof(int8_t), // Size
                                       (void *) weight_ptr, // Host ptr
                                       &ret);
        check
        // Create output buffer.
        opencl_out = clCreateBuffer(context_,
                                    CL_MEM_READ_WRITE, // Token
                                    CO * H * W * sizeof(int32_t), // Size
                                    nullptr, // Host ptr
                                    &ret);
        check
        // Record allocated cl mem
        allocated.push_back(opencl_weight);
        allocated.push_back(opencl_out);
        // Specify work dimension
        global_work_size = new size_t[3]{H, W, CO};
        local_work_size = nullptr;

    }

    void *cpu_forward(void *input) override {
        start_timer();
        // Call cpu version conv function here
        cpu_conv(CI, CO, H, W,
                 (const int8_t *) cpu_weight,
                 (const uint8_t *) input,
                 (int32_t *) cpu_out);
        cpu_time += end_timer();
        return cpu_out;
    }

    //  Set argument and execute kernel.
    void opencl_set_args(cl_mem opencl_in) override {
        // input is uint8_t
        // uint8_t *+ int8_t -> int32_t
        // Set kernel argument
        // Because the arguments never change, so specify them in constructor.
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
    int8_t *cpu_weight;

    string type() override { return "fc"; }

    fc_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
             size_t CI_, size_t CO_, int8_t *weight_ptr) :
            layer(command_queue_), CI(CI_), CO(CO_) {
        // Create kernel
        kernel = clCreateKernel(program_, "fc", &ret);
        check

        // Save cpu weight and allocate space for cpu output
        cpu_weight = weight_ptr;
        cpu_out = new int32_t[CO];

        // Create opencl_weight and result buffer;
        opencl_weight = clCreateBuffer(context_,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, // token
                                       CI * CO * sizeof(int8_t), // size
                                       (void *) weight_ptr, // host ptr
                                       &ret);
        check
        opencl_out = clCreateBuffer(context_,
                                    CL_MEM_READ_WRITE,
                                    CO * sizeof(int32_t),
                                    nullptr,
                                    &ret);
        check

        allocated.push_back(opencl_weight);
        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{CO, 1, 1};
        local_work_size = nullptr;
    }

    void opencl_set_args(cl_mem opencl_in) override {
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
    }

    void *cpu_forward(void *input) override {
        start_timer();
        cpu_fc(CI, CO, (const int8_t *) cpu_weight, (const uint8_t *) input, (int32_t *) cpu_out);
        cpu_time += end_timer();
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
    uint8_t *cpu_shift = nullptr;

    string type() override { return "quan"; }

    quan_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t C_, size_t H_, size_t W_, int32_t *bias_ptr, uint8_t *shift_ptr) :
            layer(command_queue_), C(C_), H(H_), W(W_) {
        // Create kernel
        kernel = clCreateKernel(program_, "quan", &ret);
        check

        // Save cpu bias and shift
        cpu_bias = bias_ptr;
        cpu_shift = shift_ptr;
        cpu_out = new int8_t[C * H * W];

        // Create opencl_weight and result buffer;
        opencl_bias = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, C * sizeof(int32_t),
                                     (void *) bias_ptr, &ret);
        check
        opencl_shift = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, C * sizeof(uint8_t),
                                      (void *) shift_ptr, &ret);
        check
        opencl_out = clCreateBuffer(context_, CL_MEM_READ_WRITE, C * H * W * sizeof(int8_t),
                                    nullptr, &ret);
        check

        allocated.push_back(opencl_bias);
        allocated.push_back(opencl_shift);
        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{H, W, C};
        local_work_size = nullptr;
    }

    void opencl_set_args(cl_mem opencl_in) override {
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
    }

    void *cpu_forward(void *input) override {
        start_timer();
        cpu_quan(C, H, W,
                 (const int32_t *) cpu_bias,
                 (const uint8_t *) cpu_shift,
                 (const int32_t *) input,
                 (int8_t *) cpu_out);
        cpu_time += end_timer();
        return cpu_out;
    }

    ~quan_layer() override {
        delete[] cpu_bias;
        delete[] cpu_shift;
        delete[] (int8_t *) cpu_out;
    }


};

class pool_layer : public layer {
public:
    size_t C, H, W;
    size_t HO, WO;

    string type() override { return "pool"; }

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
        cpu_out = new int8_t[C * HO * WO];

        // Create opencl_weight and result buffer;
        opencl_out = clCreateBuffer(context_,
                                    CL_MEM_READ_WRITE,
                                    C * HO * WO * sizeof(int8_t),
                                    nullptr,
                                    nullptr);

        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{HO, WO, C};
        local_work_size = nullptr;
    }

    void opencl_set_args(cl_mem opencl_in) override {
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
    }

    void *cpu_forward(void *input) override {
        start_timer();
        cpu_pool(C, H, W, HO, WO, (uint8_t *) input, (uint8_t *) cpu_out);
        cpu_time += end_timer();
        return cpu_out;
    }

    ~pool_layer() override {
        delete[] (int8_t *) cpu_out;
    }

};

class relu_layer : public layer {
public:
    size_t C, H, W;

    string type() override { return "relu"; }

    relu_layer(cl_context context_, cl_command_queue command_queue_, cl_program program_,
               size_t C_, size_t H_, size_t W_) :
            layer(command_queue_), C(C_), H(H_), W(W_) {
        // Create kernel
        kernel = clCreateKernel(program_, "relu", &ret);
        check

        // Allocate space for cpu output
        cpu_out = new uint8_t[C * H * W];

        // Create opencl_weight and result buffer;
        opencl_out = clCreateBuffer(context_,
                                    CL_MEM_READ_WRITE,
                                    C * H * W * sizeof(uint8_t),
                                    nullptr,
                                    nullptr);

        allocated.push_back(opencl_out);

        // Specify work dimension
        global_work_size = new size_t[3]{H, W, C};
        local_work_size = nullptr;
    }

    void opencl_set_args(cl_mem opencl_in) override {
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
    }

    void *cpu_forward(void *input) override {
        start_timer();
        cpu_relu(C, H, W, (int8_t *) input, (uint8_t *) cpu_out);
        cpu_time += end_timer();
        return cpu_out;
    }

    ~relu_layer() override {
        delete[] (int8_t *) cpu_out;
    }


};

class cnn {
    // Input image size, output feature size.
    size_t IMAGE_C, IMAGE_H, IMAGE_W, FEATURE;
    // Opencl related variables.
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;

    // Create opencl input buffer
    // set a output buffer for both opencl and cpu inference
    cl_mem opencl_in = nullptr;
    int8_t *out_buff = nullptr;

    // Container of layers.
    vector<layer *> layers;

public:
    void report_cpu_time() {
        cout << "********************" << endl;
        for (auto &layer:layers)layer->report_cpu_time();
        map<string, double> cpu_time_table;
        for (auto &layer:layers) cpu_time_table[layer->type()] = 0;
        for (auto &layer:layers) cpu_time_table[layer->type()] += layer->cpu_time;
        double total_time;
        for (auto &p:cpu_time_table) {
            cout << "Total " << p.first << " time: " << p.second << endl;
            total_time += p.second;
        }
        cout << "Total CNN time: " << total_time << endl;
        cout << "********************" << endl;
    }

    void report_opencl_time() {
        cout << "********************" << endl;
        for (auto &layer:layers)layer->report_opencl_time();
        map<string, double> opencl_time_table;
        for (auto &layer:layers) opencl_time_table[layer->type()] += 0;
        for (auto &layer:layers) opencl_time_table[layer->type()] += layer->opencl_time;
        double total_time = 0;
        for (auto &p:opencl_time_table) {
            cout << "Total " << p.first << " time: " << p.second << endl;
            total_time += p.second;
        }
        cout << "Total CNN time: " << total_time << endl;
        cout << "********************" << endl;
    }

    void parse_model_file(const string &model_file) {
        ifstream fs(model_file);
        string s;
        int param;
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
                auto weight_ptr = new int8_t[CO * CI * 3 * 3];
                for (int co = 0; co < CO; co++) {
                    for (int ci = 0; ci < CI; ci++) {
                        for (int h = 0; h < 3; h++) {
                            for (int w = 0; w < 3; w++) {
                                fs >> param;
                                weight_ptr[co * CI * 3 * 3 + ci * 3 * 3 + h * 3 + w] = param;
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
                auto weight_ptr = new int8_t[CI * CO];
                for (int ci = 0; ci < CI; ci++) {
                    for (int co = 0; co < CO; co++) {
                        fs >> param;
                        weight_ptr[ci * CO + co] = param;
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
                auto shift_ptr = new uint8_t[C];
                fs >> s;
                assert(s == "BIAS");
                for (int c = 0; c < C; c++) {
                    fs >> param;
                    bias_ptr[c] = param;
                }
                fs >> s;
                assert(s == "SHIFT");
                for (int c = 0; c < C; c++) {
                    fs >> param;
                    shift_ptr[c] = param;
                }
                layers.emplace_back(new quan_layer(context, command_queue, program, C, H, W, bias_ptr, shift_ptr));
            } else {
                cout << "No such layer: " << s << endl;
                exit(1);
            }
        }
    }

    cnn(size_t C_, size_t H_, size_t W_, size_t FEATURE_, const string &kernel_file, const string &model_file) :
            IMAGE_C(C_), IMAGE_H(H_), IMAGE_W(W_), FEATURE(FEATURE_) {
        opencl_init(kernel_file);
        parse_model_file(model_file);
        out_buff = new int8_t[FEATURE];
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
        opencl_in = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY, // Token
                                   IMAGE_C * IMAGE_H * IMAGE_W * sizeof(uint8_t), // buffer size
                                   nullptr,  // host ptr
                                   &ret);
        check
    }

    void opencl_release() {
        // Release.
        // Kernels will be released in the deconstruct function of layers
        clReleaseMemObject(opencl_in);
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

    size_t opencl_forward(uint8_t *image) {
        ret = clEnqueueWriteBuffer(command_queue,
                                   opencl_in,
                                   CL_FALSE,  // Block writing. If blocking, this function will finish queue.
                                   0, // Offset
                                   IMAGE_C * IMAGE_H * IMAGE_W * sizeof(uint8_t), // Size
                                   image,
                                   0,  // wait number
                                   nullptr, // wait list
                                   nullptr); // bounding event
        check
        cl_mem cur = opencl_in;
        for (auto layer:layers) cur = layer->opencl_forward(cur);
        ret = clEnqueueReadBuffer(command_queue,
                                  cur,
                                  CL_TRUE, // Block reading. Finish queue and read.
                                  0, // offset
                                  FEATURE * sizeof(int8_t), // read size
                                  out_buff,
                                  0,
                                  nullptr,
                                  nullptr);
        check
        return argmax(out_buff, FEATURE);
    }

    size_t cpu_forward(uint8_t *image) {
        void *cur = image;
        for (auto &layer : layers)cur = layer->cpu_forward(cur);
        return argmax((int8_t *) cur, FEATURE);
    }
};


#endif