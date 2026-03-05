

#ifndef MC_CUH
#define MC_CUH

#define N_PATHS (1024 * 1024)
#define NTPB 1024
#define NB ((N_PATHS + NTPB - 1) / NTPB)
#define N_STEPS 1000
#define T_FINAL 10.0f

const float host_a = 1.0f;
const float host_sigma = 0.1f;
const float host_r0 = 0.012f;
const float host_dt = T_FINAL / N_STEPS; // 0.01

__constant__ float device_a;
__constant__ float device_sigma;
__constant__ float device_r0;
__constant__ float device_dt;
__constant__ float device_mean_reversion_factor;
__constant__ float device_std_gaussian_shock;
__constant__ float device_drift_table[N_STEPS];

void init_device_constants(){
    float host_mean_reversion_factor = expf(-host_a * host_dt);
    float host_std_gaussian_shock = host_sigma * sqrtf((1.0f- expf(-2.0f*host_a*host_dt))/(2.0f*host_a));
    
    // to start simple we assume a flat curve which leads to same drift term for each time step
    // therefore the drift always pulls r back to r0
    float host_drift_term = host_r0 * (1.0f - host_mean_reversion_factor);
    float host_drift_table[N_STEPS];

    for(int i=0; i< N_STEPS; i++){
        host_drift_table[i] = host_drift_term;
    }

    cudaMemcpyToSymbol(device_a, &host_a, sizeof(float));
    cudaMemcpyToSymbol(device_sigma, &host_sigma, sizeof(float));
    cudaMemcpyToSymbol(device_r0, &host_r0, sizeof(float));
    cudaMemcpyToSymbol(device_dt, &host_dt, sizeof(float));
    cudaMemcpyToSymbol(device_mean_reversion_factor, &host_mean_reversion_factor, sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &host_std_gaussian_shock, sizeof(float));
    cudaMemcpyToSymbol(device_drift_table, host_drift_table, N_STEPS * sizeof(float));

}

#endif // MC_CUH