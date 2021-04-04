#include <bits/stdc++.h>
#include <CL/cl.h>

using namespace std;


const int MAX_LEN = 1024 * 1024 * 64; // Max work-item number for GTX 1070.
int A[MAX_LEN], B[MAX_LEN], C_GPU[MAX_LEN], C_CPU[MAX_LEN], C_STD[MAX_LEN];

string readProgramFile(const string &file_path) {
    ifstream ifs(file_path);
    stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

void opencl_vec_add(int n) {
    cl_int ret_status;
    cl_platform_id platform;
    cl_device_id device;
    ret_status = clGetPlatformIDs(1, &platform, nullptr);
    ret_status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret_status);
    auto command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret_status);

    size_t buff_size = n * sizeof(int);
    auto buff_a = clCreateBuffer(context, CL_MEM_READ_ONLY, buff_size, nullptr, &ret_status);
    auto buff_b = clCreateBuffer(context, CL_MEM_READ_ONLY, buff_size, nullptr, &ret_status);
    auto buff_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buff_size, nullptr, &ret_status);

    ret_status = clEnqueueWriteBuffer(command_queue, buff_a, CL_FALSE, 0, buff_size, A, 0, nullptr, nullptr);
    ret_status = clEnqueueWriteBuffer(command_queue, buff_b, CL_FALSE, 0, buff_size, B, 0, nullptr, nullptr);

    string src = readProgramFile("../vec_add.cl");
    auto src_buff = new char[src.size() + 1];
    src_buff[src.size()] = 0;
    memcpy(src_buff, src.c_str(), sizeof(char) * src.size());
    auto program = clCreateProgramWithSource(context, 1, (const char **) &src_buff, nullptr, &ret_status);
    ret_status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    auto kernel = clCreateKernel(program, "vec_add", &ret_status);

    int work_item_len;
    if (n > 1024 * 1024 * 64)work_item_len = n / (1024 * 1024 * 64);
    else work_item_len = 1;
    ret_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buff_a) |
                 clSetKernelArg(kernel, 1, sizeof(cl_mem), &buff_b) |
                 clSetKernelArg(kernel, 2, sizeof(cl_mem), &buff_c);

    size_t work_size[3];
    if (n <= 1024) {
        work_size[0] = n;
        work_size[1] = 1;
        work_size[2] = 1;
    } else if (n <= 1024 * 1024) {
        work_size[0] = 1024;
        work_size[1] = n / 1024;
        work_size[2] = 1;
    } else if (n <= 1024 * 1024 * 64) {
        work_size[0] = 1024;
        work_size[1] = 1024;
        work_size[2] = n / 1024 / 1024;
    } else {
        cout << "N too big." << endl;
        assert(1);
        return;
    }

    clFinish(command_queue);
    cl_event kernel_exec_event;

    ret_status = clEnqueueNDRangeKernel(command_queue, kernel, 3, nullptr, work_size, nullptr, 0, nullptr,
                                        &kernel_exec_event);
    clWaitForEvents(1, &kernel_exec_event);

    cl_ulong op, ed;
    clGetEventProfilingInfo(kernel_exec_event, CL_PROFILING_COMMAND_START, sizeof(op), &op, nullptr);
    clGetEventProfilingInfo(kernel_exec_event, CL_PROFILING_COMMAND_END, sizeof(ed), &ed, nullptr);
    cout << "Opencl time: " << double(ed - op) / 1e9 << "s" << endl;

    clEnqueueReadBuffer(command_queue, buff_c, CL_TRUE, 0, buff_size, C_GPU, 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(buff_a);
    clReleaseMemObject(buff_b);
    clReleaseMemObject(buff_c);
    clReleaseContext(context);
}

void cpu_vec_add(int N) {
    using namespace chrono;
    auto op = system_clock::now();

    for (int i = 0; i < N; ++i) {
        C_CPU[i] = A[i] + B[i];
    }
    auto ed = system_clock::now();
    auto t = ed - op;
    cout << "CPU time: " << t.count() / 1e9 << "s" << endl;
}

int main() {
    // Get input data.
    const int RANDOM_SEED = time(nullptr);
    default_random_engine r(RANDOM_SEED);
    for (int i = 0; i < MAX_LEN; i++) {
        A[i] = r(), B[i] = r();
        C_STD[i] = A[i] + B[i];
    }

    for (int len = 1024; len <= 1024 * 1024 * 64; len *= 2) {
        cout << "**********LEN: " << setw(10) << len << "**********" << endl;
        bool flag = true;

        cpu_vec_add(len);
        flag = true;
        for (int i = 0; i < len; i++) {
            if (C_CPU[i] != C_STD[i]) {
                flag = false;
                break;
            }
        }
        if (flag)cout << "CPU correct." << endl;
        else cout << "CPU incorrect." << endl;

        opencl_vec_add(len);
        for (int i = 0; i < len; i++) {
            if (C_GPU[i] != C_STD[i]) {
                flag = false;
                break;
            }
        }
        if (flag) cout << "Opencl correct." << endl;
        else cout << "Opencl incorrect." << endl;

        cout << "************************" << endl;
    }

    return 0;
}
