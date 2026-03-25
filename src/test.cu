#include "mc_calibration.cuh"
#include "mc_market_price.cuh"
#include "mc_swaptions.cuh"
#include "mc_option_pricing.cuh"
#include "hw_swaptions.cuh"
#include <cstdio>

const float a     = 1.0f;
const float sigma = 0.1f;
const float r0    = 0.012f;

inline void price_swaption(float* h_out, curandState* d_states,
                            const float* d_P0, const float* d_f0,
                            float T, float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, 2 * sizeof(float));
    cudaMemset(d_out,  0, 2 * sizeof(float));
    simulate_swaption<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, a, sigma, r0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    h_out[1] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_delta(float* h_out, curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  float T, float a, float sigma, float r0){
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    simulate_swaption_delta<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, a, sigma, r0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS;
    cudaFree(d_out);
}

inline void price_swaption_gamma(float* h_out, curandState* d_states,
                                  const float* d_P0, const float* d_f0,
                                  float T, float a, float sigma, float r0, float eps){
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out,  0, sizeof(float));
    simulate_swaption_gamma<<<NB, NTPB>>>(d_out, d_states, d_P0, d_f0, T, a, sigma, r0, eps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    h_out[0] /= N_PATHS * eps * eps;
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

    float tenor_dates[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int   n_tenors      = 5;
    float eps           = 0.001f;

    float K_swap = par_swap_rate(1.0f, tenor_dates, n_tenors, h_P);
    float c[5];
    for(int i = 0; i < n_tenors - 1; i++) c[i] = K_swap;
    c[n_tenors - 1] = 1.0f + K_swap;

    float price = analytical_swaption      (1.0f, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float vega  = analytical_swaption_vega (1.0f, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float delta = analytical_swaption_delta(1.0f, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);
    float gamma = analytical_swaption_gamma(1.0f, tenor_dates, n_tenors, c, h_P, f0, a, sigma, r0);

    init_swaption(tenor_dates, c, n_tenors);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_price[2];
    price_swaption(h_price, d_states, d_P0, d_f0, 1.0f, a, sigma, r0);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_delta[1];
    price_swaption_delta(h_delta, d_states, d_P0, d_f0, 1.0f, a, sigma, r0);

    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    float h_gamma[1];
    price_swaption_gamma(h_gamma, d_states, d_P0, d_f0, 1.0f, a, sigma, r0, eps);

    printf("\n=== Swaption Pricing ===\n");
    printf("Par swap rate K : %.6f\n", K_swap);
    printf("%-16s  %-12s  %-12s  %-12s\n", "", "MC", "Analytical", "Error");
    printf("%-16s  %-12.6f  %-12.6f  %-12.2e\n",
           "Price", h_price[0], price, fabsf(h_price[0] - price));
    printf("%-16s  %-12.6f  %-12.6f  %-12.2e\n",
           "Vega",  h_price[1], vega,  fabsf(h_price[1] - vega));
    printf("%-16s  %-12.6f  %-12.6f  %-12.2e\n",
           "Delta", h_delta[0], delta, fabsf(h_delta[0] - delta));
    printf("%-16s  %-12.6f  %-12.6f  %-12.2e\n",
           "Gamma", h_gamma[0], gamma, fabsf(h_gamma[0] - gamma));

    cudaFree(d_P0);
    cudaFree(d_f0);
    cudaFree(d_states);
    return 0;
}