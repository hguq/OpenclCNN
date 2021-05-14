//
// Created by hiro on 2021/5/13.
//

#ifndef OPENCL_CNN_CONV_TIMER_CPP
#define OPENCL_CNN_CONV_TIMER_CPP

#include <chrono>

using namespace std::chrono;

static auto op = system_clock::now();
static auto ed = system_clock::now();


void start_timer() {
    op = system_clock::now();
}

//return in second
double end_timer() {
    ed = system_clock::now();
    auto t = (ed - op).count() * 1.0 / 1e9;
    return t;
}


#endif //OPENCL_CNN_CONV_TIMER_CPP
