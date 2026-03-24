#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "hw_option_pricing.cuh"
#include "hw_option_sensitivities.cuh"
#include "mc_option_pricing.cuh"
#include <cstdio>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

inline void price_option(float* h_out,
                          curandState* d_states,
                          const float* d_P0, const float* d_f0,
                          float T, float S, float X,
                          float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, 6 * sizeof(float));
    cudaMemset(d_out,  0, 6 * sizeof(float));

    simulate_option<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0,
                                   T, S, X, a, sigma, r0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 6; i++) h_out[i] /= N_PATHS;

    cudaFree(d_out);
}

int main(){
    init(a, sigma);

    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    init_drift(a, sigma, r0);

    float h_P[N_MAT];
    simulate_market_price(h_P, d_states, r0);

    float f0[N_MAT];
    calibrate(h_P, f0, a, sigma);

    float* d_P0;
    float* d_f0;
    cudaMalloc(&d_P0, N_MAT * sizeof(float));
    cudaMalloc(&d_f0, N_MAT * sizeof(float));
    cudaMemcpy(d_P0, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f0, f0,  N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    float T = 1.0f;
    float S = 5.0f;
    float X = expf(-r0 * S);

    EuroOption o              = euro_option(h_P, f0, 0.0f, T, S, X, r0, a, sigma);
    float analytical_zbc      = ZBC(o);
    float analytical_zbp      = ZBP(o);
    float analytical_vega_zbc = vega_zbc(o, 0.0f, T, S, a, sigma);
    float analytical_vega_zbp = vega_zbp(o, 0.0f, T, S, a, sigma);
    float analytical_volga_zbc = volga_zbc(o, 0.0f, T, S, a, sigma);
    float analytical_volga_zbp = volga_zbp(o, 0.0f, T, S, a, sigma);

    float h_out[6];
    price_option(h_out, d_states, d_P0, d_f0, T, S, X, a, sigma, r0);

    printf("\n=== Option Pricing ===\n");
    printf("ZBC       MC: %.6f  |  Analytical: %.6f  |  Error: %.2e\n",
           h_out[0], analytical_zbc,       fabsf(h_out[0] - analytical_zbc));
    printf("ZBP       MC: %.6f  |  Analytical: %.6f  |  Error: %.2e\n",
           h_out[1], analytical_zbp,       fabsf(h_out[1] - analytical_zbp));
    printf("Vega  ZBC MC: %.6f  |  Analytical: %.6f  |  Error: %.2e\n",
           h_out[2], analytical_vega_zbc,  fabsf(h_out[2] - analytical_vega_zbc));
    printf("Vega  ZBP MC: %.6f  |  Analytical: %.6f  |  Error: %.2e\n",
           h_out[3], analytical_vega_zbp,  fabsf(h_out[3] - analytical_vega_zbp));
    printf("Volga ZBC MC: %.6f  |  Analytical: %.6f  |  Error: %.2e\n",
           h_out[4], analytical_volga_zbc, fabsf(h_out[4] - analytical_volga_zbc));
    printf("Volga ZBP MC: %.6f  |  Analytical: %.6f  |  Error: %.2e\n",
           h_out[5], analytical_volga_zbp, fabsf(h_out[5] - analytical_volga_zbp));

    cudaFree(d_P0);
    cudaFree(d_f0);
    cudaFree(d_states);
    return 0;
}