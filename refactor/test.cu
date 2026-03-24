#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include <cstdio>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

int main(){
    init(a, sigma);

    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    init_drift(a, sigma, r0);

    float h_P[N_MAT];
    simulate_market_price(h_P, d_states, r0);

    printf("h_P[9]=%.6f h_P[49]=%.6f h_P[99]=%.6f\n",
           h_P[9], h_P[49], h_P[99]);

    cudaFree(d_states);
    return 0;
}