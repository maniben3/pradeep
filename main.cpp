/* main.cpp */

#include <iostream>
#include <stdint.h>
#include <math.h>                   // pow()
#include <cuda_runtime.h>           // cudaFreeHost()
#include "CUDASieve/cudasieve.hpp"  // CudaSieve::getHostPrimes()

int main()
{
    uint64_t bottom = pow(2,63);
    uint64_t top = pow(2,63)+pow(2,30);
    size_t len;

    uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len);

    for(uint32_t i = 0; i < len; i++)
        std::cout << primes[i] << std::endl;

    cudaFreeHost(primes);            // must be freed with this call b/c page-locked memory is used.
    return 0;
}
