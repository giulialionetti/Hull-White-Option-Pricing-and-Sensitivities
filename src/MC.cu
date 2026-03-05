#include "mc.cuh"
#include "logger.h"


void analytical_greeks(float t, float T, float S, float K, float rt){
    float zbc       = ZBC(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float zbp       = ZBP(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float parity    = P0T(S, host_r0) - K * P0T(T, host_r0);
    float vega_zbc  = vega_ZBC(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float vega_zbp  = vega_ZBP(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float delta_zbc = delta_ZBC(t, T, S, K, rt, host_a, host_sigma, host_r0);
    float delta_zbp = delta_ZBP(t, T, S, K, rt, host_a, host_sigma, host_r0);

    float parity_err = (zbc - zbp) - parity;

    LOG_INFO("=== Analytical Pricing ===");
    LOG_INFO("Params: t=%.2f  T=%.2f  S=%.2f  K=%.6f  r0=%.6f", t, T, S, K, rt);
    LOG_INFO("ZBC               : %.6f", zbc);
    LOG_INFO("ZBP               : %.6f", zbp);
    LOG_INFO("ZBC - ZBP         : %.6f", zbc - zbp);
    LOG_INFO("P(0,S) - K*P(0,T) : %.6f", parity);
    if (fabsf(parity_err) < 1e-5f)
        LOG_INFO("Put-call parity   : OK   (err=%.2e)", parity_err);
    else
        LOG_WARN("Put-call parity   : FAIL (err=%.2e)", parity_err);
    LOG_INFO("Vega  ZBC         : %.6f", vega_zbc);
    LOG_INFO("Delta ZBC         : %.6f", delta_zbc);
    LOG_INFO("Vega  ZBP         : %.6f", vega_zbp);
    LOG_INFO("Delta ZBP         : %.6f", delta_zbp);
}

void monteCarlo_vega(float T, float S, float K, curandState* d_states, CurveType curve){
   
    float* d_ZBC   = nullptr;
    float* d_vega  = nullptr;
    cudaMalloc(&d_ZBC,  sizeof(float));
    cudaMalloc(&d_vega, sizeof(float));
    cudaMemset(d_ZBC,  0, sizeof(float));
    cudaMemset(d_vega, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const char* label = (curve == CurveType::FLAT) ? "FLAT" : "PIECEWISE";
    LOG_INFO("=== MC Pricing [%s curve] ===", label);

    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC, d_vega, d_states, T, S, K);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float h_ZBC, h_vega;
    cudaMemcpy(&h_ZBC,  d_ZBC,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vega, d_vega, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC  /= N_PATHS;
    h_vega /= N_PATHS;

    float analytical_zbc  = ZBC(0.0f, T, S, K, host_r0, host_a, host_sigma, host_r0);
    float analytical_vega = vega_ZBC(0.0f, T, S, K, host_r0, host_a, host_sigma, host_r0);

    LOG_INFO("MC ZBC:          %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_ZBC, analytical_zbc, fabsf(h_ZBC - analytical_zbc));
    LOG_INFO("MC Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
             h_vega, analytical_vega, fabsf(h_vega - analytical_vega));
    LOG_INFO("Simulation time: %.2f ms  |  Throughput: %.2f M paths/sec",
             elapsed_ms, N_PATHS / elapsed_ms / 1000.0f);

    cudaFree(d_ZBC);
    cudaFree(d_vega);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

void finitedifferences_mc_vega(float T, float S, float K, curandState* d_states, CurveType curve){

    float eps  = 0.001f;
    unsigned long seed = time(NULL);

    float* d_ZBC_plus  = nullptr;
    float* d_ZBC_minus = nullptr;
    float* d_vega_dummy = nullptr;
    cudaMalloc(&d_ZBC_plus,   sizeof(float));
    cudaMalloc(&d_ZBC_minus,  sizeof(float));
    cudaMalloc(&d_vega_dummy, sizeof(float));

    // bump up
    cudaMemset(d_ZBC_plus,  0, sizeof(float));
    cudaMemset(d_vega_dummy, 0, sizeof(float));
    init_device_constants(host_sigma + eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_plus, d_vega_dummy, d_states, T, S, K);
    cudaDeviceSynchronize();

    // bump down — same seed
    cudaMemset(d_ZBC_minus,  0, sizeof(float));
    cudaMemset(d_vega_dummy, 0, sizeof(float));
    init_device_constants(host_sigma - eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_minus, d_vega_dummy, d_states, T, S, K);
    cudaDeviceSynchronize();

    float h_plus, h_minus;
    cudaMemcpy(&h_plus,  d_ZBC_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_ZBC_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_minus /= N_PATHS;

    float vega_fd         = (h_plus - h_minus) / (2.0f * eps);
    float analytical_vega = vega_ZBC(0.0f, T, S, K, host_r0, host_a, host_sigma, host_r0);

    LOG_INFO("FD Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
             vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));

    cudaFree(d_ZBC_plus);
    cudaFree(d_ZBC_minus);
    cudaFree(d_vega_dummy);

}

int main(){
    
    Logger& log = Logger::instance();
    log.open_file("hw_output.log");
    log.set_level(LogLevel::DEBUG);

    float t = 0.0f;
    float T = 1.0f;
    float S = 5.0f;
    float K = P0T(S, host_r0);

    analytical_greeks(t, T, S, K, host_r0);

    init_device_constants();
    curandState* d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    init_device_constants(host_sigma, CurveType::FLAT);
    monteCarlo_vega(T, S, K, d_states, CurveType::FLAT);
    finitedifferences_mc_vega(T, S, K, d_states,CurveType::FLAT);

    init_device_constants(host_sigma, CurveType::PIECEWISE_LINEAR);
    monteCarlo_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR);
    finitedifferences_mc_vega(T, S, K, d_states, CurveType::PIECEWISE_LINEAR);

    cudaFree(d_states);
    return 0;
}