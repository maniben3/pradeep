#include <iostream>

using namespace std ;

# define DELLEXPORT extern "C" __declspec(dllexport)

__global__ void kernel(long* answer = 0){
    *answer = threadIdx.x + (blockIdx.x * blockDim.x);
}

DELLEXPORT void resoult(long* h_answer){

    long* d_answer = 0;
    
    cudaMalloc(&d_answer, sizeof(long));

    kernel<<<10,1000>>>(d_answer);
    cudaMemcpy(&h_answer, d_answer, sizeof(long), cudaMemcpyDeviceToHost);
        cudaFree(d_answer);
}
