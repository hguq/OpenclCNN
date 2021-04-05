__kernel void vec_add(__global int *A, __global int *B, __global int *C) {
   int idx =    get_global_id(0) * get_global_size(1) * get_global_size(2) +
                get_global_id(1) * get_global_size(2) +
                get_global_id(2);
   C[idx] = A[idx] + B[idx];
 }

