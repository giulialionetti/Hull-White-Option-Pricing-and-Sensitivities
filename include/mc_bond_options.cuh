#ifndef MC_BOND_OPTIONS_CUH
#define MC_BOND_OPTIONS_CUH

#include "mc_engine.cuh"
#include "hw_pricing.cuh"
#include "hw_greeks_first.cuh"
#include "hw_greeks_second.cuh"

__global__ void mc_zbc_vega(float* ZBC_estimator, float* vega_estimator,
                             float* volga_estimator,
                             curandState* states, float T_maturity, float S, float K,
                             const float* P_market, const float* f_market){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared_zbc[NTPB];
    __shared__ float shared_vega[NTPB];
    __shared__ float shared_volga[NTPB];

    float thread_zbc   = 0.0f;
    float thread_vega  = 0.0f;
    float thread_volga = 0.0f;

    if(path_id < N_PATHS){
        curandState local_state = states[path_id];

        float r_step_i                 = device_r0;
        float discount_factor_integral = 0.0f;
        float drdsigma_step_i          = 0.0f;
        float drdsigma_integral        = 0.0f;

        int n_steps_T = (int)(T_maturity / device_dt);
        for(int i = 0; i < n_steps_T; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r_step_i, discount_factor_integral,
                              device_drift_table[i], G);
            evolve_short_rate_derivative(drdsigma_step_i, drdsigma_integral,
                                        device_sensitivity_drift_table[i], G);
        }

        float bond_price = (P_market == nullptr) ?
            FlatCurve{device_a, device_sigma, device_r0}.P(T_maturity, S, r_step_i) :
            MarketCurve{device_a, device_sigma, P_market, f_market,
                        MAT_SPACING, N_MAT}.P(T_maturity, S, r_step_i);

        float discount_factor  = expf(-discount_factor_integral);
       float B_val          = BtT(T_maturity, S, device_a);
float dPricedsigma   = -bond_price * B_val * drdsigma_step_i;
float d2Pricedsigma2 =  bond_price * B_val * B_val
                      * drdsigma_step_i * drdsigma_step_i;

        float payoff           = fmaxf(bond_price - K, 0.0f);
        float in_the_money     = (bond_price > K) ? 1.0f : 0.0f;

        thread_zbc   = discount_factor * payoff;
        thread_vega  = discount_factor * dPricedsigma * in_the_money
                     - drdsigma_integral * discount_factor * payoff;
        thread_volga = drdsigma_integral * drdsigma_integral * discount_factor * payoff
             - 2.0f * drdsigma_integral * discount_factor * dPricedsigma * in_the_money
             + discount_factor * d2Pricedsigma2 * in_the_money;
        states[path_id] = local_state;
    }

    shared_zbc[threadIdx.x]   = thread_zbc;
    shared_vega[threadIdx.x]  = thread_vega;
    shared_volga[threadIdx.x] = thread_volga;
    __syncthreads();

    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            shared_zbc[threadIdx.x]   += shared_zbc[threadIdx.x + i];
            shared_vega[threadIdx.x]  += shared_vega[threadIdx.x + i];
            shared_volga[threadIdx.x] += shared_volga[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(ZBC_estimator,   shared_zbc[0]);
        atomicAdd(vega_estimator,  shared_vega[0]);
        atomicAdd(volga_estimator, shared_volga[0]);
    }
}


void monteCarlo_vega(float T, float S, float K, curandState* d_states, CurveType curve,
                     float* d_P_market = nullptr, float* d_f_market = nullptr,
                     float* h_P = nullptr, float* h_f = nullptr){

    float* d_ZBC   = nullptr;
    float* d_vega  = nullptr;
    float* d_volga = nullptr;
    cudaMalloc(&d_ZBC,   sizeof(float));
    cudaMalloc(&d_vega,  sizeof(float));
    cudaMalloc(&d_volga, sizeof(float));
    cudaMemset(d_ZBC,   0, sizeof(float));
    cudaMemset(d_vega,  0, sizeof(float));
    cudaMemset(d_volga, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const char* label = (curve == CurveType::FLAT)  ? "FLAT"      :
                        (d_P_market == nullptr)       ? "PIECEWISE" : "PIECEWISE+MARKET";
    LOG_INFO("=== MC Pricing [%s curve] ===", label);

    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC, d_vega, d_volga, d_states, T, S, K,
                               d_P_market, d_f_market);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float h_ZBC, h_vega, h_volga;
    cudaMemcpy(&h_ZBC,   d_ZBC,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vega,  d_vega,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_volga, d_volga, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC   /= N_PATHS;
    h_vega  /= N_PATHS;
    h_volga /= N_PATHS;

    if(d_P_market == nullptr){
        auto ps           = make_pricing_state(0.0f, T, S, K, host_r0,
                                               host_a, host_sigma,
                                               FlatCurve{host_a, host_sigma, host_r0});
        float analytical_zbc   = ZBC_from_state(ps);
        float analytical_vega  = vega_ZBC_from_state(ps);
        float analytical_volga = volga_ZBC_from_state(ps);
        LOG_INFO("MC ZBC:          %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 h_ZBC,   analytical_zbc,   fabsf(h_ZBC   - analytical_zbc));
        LOG_INFO("MC Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 h_vega,  analytical_vega,  fabsf(h_vega  - analytical_vega));
        LOG_INFO("MC Volga:        %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 h_volga, analytical_volga, fabsf(h_volga - analytical_volga));
    } else {
        auto ps           = make_pricing_state(0.0f, T, S, K, host_r0,
                                               host_a, host_sigma,
                                               MarketCurve{host_a, host_sigma,
                                                           h_P, h_f,
                                                           MAT_SPACING, N_MAT});
        float analytical_zbc   = ZBC_from_state(ps);
        float analytical_vega  = vega_ZBC_from_state(ps);
        float analytical_volga = volga_ZBC_from_state(ps);
        LOG_INFO("MC ZBC:          %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 h_ZBC,   analytical_zbc,   fabsf(h_ZBC   - analytical_zbc));
        LOG_INFO("MC Vega:         %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 h_vega,  analytical_vega,  fabsf(h_vega  - analytical_vega));
        LOG_INFO("MC Volga:        %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 h_volga, analytical_volga, fabsf(h_volga - analytical_volga));
    }

    cudaFree(d_ZBC);
    cudaFree(d_vega);
    cudaFree(d_volga);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void finitedifferences_mc_vega(float T, float S, float K, curandState* d_states,
                                CurveType curve,
                                float* d_P_market = nullptr, float* d_f_market = nullptr,
                                float* h_P = nullptr, float* h_f = nullptr){

    float eps          = 0.1f;
    unsigned long seed = time(NULL);

    float* d_ZBC_plus   = nullptr;
    float* d_ZBC_minus  = nullptr;
    float* d_vega_dummy = nullptr;
    float* d_volga_dummy = nullptr;
    cudaMalloc(&d_ZBC_plus,    sizeof(float));
    cudaMalloc(&d_ZBC_minus,   sizeof(float));
    cudaMalloc(&d_vega_dummy,  sizeof(float));
    cudaMalloc(&d_volga_dummy, sizeof(float));

    cudaMemset(d_ZBC_plus,    0, sizeof(float));
    cudaMemset(d_vega_dummy,  0, sizeof(float));
    cudaMemset(d_volga_dummy, 0, sizeof(float));
    init_device_constants(host_sigma + eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_plus, d_vega_dummy, d_volga_dummy,
                               d_states, T, S, K, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    cudaMemset(d_ZBC_minus,   0, sizeof(float));
    cudaMemset(d_vega_dummy,  0, sizeof(float));
    cudaMemset(d_volga_dummy, 0, sizeof(float));
    init_device_constants(host_sigma - eps, curve);
    init_rng<<<NB, NTPB>>>(d_states, seed);
    cudaDeviceSynchronize();
    mc_zbc_vega<<<NB, NTPB>>>(d_ZBC_minus, d_vega_dummy, d_volga_dummy,
                               d_states, T, S, K, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    float h_plus, h_minus;
    cudaMemcpy(&h_plus,  d_ZBC_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_ZBC_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_minus /= N_PATHS;

    float vega_fd = (h_plus - h_minus) / (2.0f * eps);

    if(d_P_market == nullptr){
        auto ps               = make_pricing_state(0.0f, T, S, K, host_r0,
                                                   host_a, host_sigma,
                                                   FlatCurve{host_a, host_sigma, host_r0});
        float analytical_vega = vega_ZBC_from_state(ps);
        LOG_INFO("FD Vega:         %.6f  |  Analytical: %.6f  |  Error: %.2e",
                 vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));
    } else {
        auto ps               = make_pricing_state(0.0f, T, S, K, host_r0,
                                                   host_a, host_sigma,
                                                   MarketCurve{host_a, host_sigma,
                                                               h_P, h_f,
                                                               MAT_SPACING, N_MAT});
        float analytical_vega = vega_ZBC_from_state(ps);
        LOG_INFO("FD Vega:         %.6f  |  Analytical (market): %.6f  |  Error: %.2e",
                 vega_fd, analytical_vega, fabsf(vega_fd - analytical_vega));
    }

    cudaFree(d_ZBC_plus);
    cudaFree(d_ZBC_minus);
    cudaFree(d_vega_dummy);
    cudaFree(d_volga_dummy);
}


void finitedifferences_mc_zbc_volga(float T, float S, float K,
                                     curandState* d_states, CurveType curve,
                                     float* d_P_market = nullptr,
                                     float* d_f_market = nullptr){

    float eps          = 0.01f;
    unsigned long seed = time(NULL);

    float* d_zbc_plus  = nullptr;
    float* d_zbc_mid   = nullptr;
    float* d_zbc_minus = nullptr;
    float* d_dummy_v   = nullptr;
    float* d_dummy_vg  = nullptr;
    cudaMalloc(&d_zbc_plus,  sizeof(float));
    cudaMalloc(&d_zbc_mid,   sizeof(float));
    cudaMalloc(&d_zbc_minus, sizeof(float));
    cudaMalloc(&d_dummy_v,   sizeof(float));
    cudaMalloc(&d_dummy_vg,  sizeof(float));

    auto run_pass = [&](float sig, float* d_out){
        float shock = sig * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
        cudaMemset(d_out,      0, sizeof(float));
        cudaMemset(d_dummy_v,  0, sizeof(float));
        cudaMemset(d_dummy_vg, 0, sizeof(float));
        cudaMemcpyToSymbol(device_sigma,              &sig,   sizeof(float));
        cudaMemcpyToSymbol(device_std_gaussian_shock, &shock, sizeof(float));
        init_rng<<<NB, NTPB>>>(d_states, seed);
        cudaDeviceSynchronize();
        mc_zbc_vega<<<NB, NTPB>>>(d_out, d_dummy_v, d_dummy_vg,
                                   d_states, T, S, K, d_P_market, d_f_market);
        cudaDeviceSynchronize();
    };

    run_pass(host_sigma + eps, d_zbc_plus);
    run_pass(host_sigma,       d_zbc_mid);
    run_pass(host_sigma - eps, d_zbc_minus);

    float shock_orig = host_sigma * sqrtf((1.0f - expf(-2.0f*host_a*host_dt)) / (2.0f*host_a));
    cudaMemcpyToSymbol(device_sigma,              &host_sigma, sizeof(float));
    cudaMemcpyToSymbol(device_std_gaussian_shock, &shock_orig, sizeof(float));

    float h_plus, h_mid, h_minus;
    cudaMemcpy(&h_plus,  d_zbc_plus,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mid,   d_zbc_mid,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minus, d_zbc_minus, sizeof(float), cudaMemcpyDeviceToHost);
    h_plus  /= N_PATHS;
    h_mid   /= N_PATHS;
    h_minus /= N_PATHS;

    LOG_INFO("FD Volga:        %.6f",
             (h_plus - 2.0f * h_mid + h_minus) / (eps * eps));

    cudaFree(d_zbc_plus);
    cudaFree(d_zbc_mid);
    cudaFree(d_zbc_minus);
    cudaFree(d_dummy_v);
    cudaFree(d_dummy_vg);
}

__global__ void mc_zbc_delta_gamma(float* ZBC_estimator,
                                    float* delta_estimator,
                                    float* gamma_estimator,
                                    curandState* states,
                                    float T_maturity, float S, float K,
                                    float epsilon,
                                    const float* P_market, const float* f_market){

    int path_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared_zbc  [NTPB];
    __shared__ float shared_delta[NTPB];
    __shared__ float shared_gamma[NTPB];

    float thread_zbc   = 0.0f;
    float thread_delta = 0.0f;
    float thread_gamma = 0.0f;

    if(path_id < N_PATHS){
        curandState local_state = states[path_id];

        float r_step_i                 = device_r0;
        float discount_factor_integral = 0.0f;

        int n_steps_T = (int)(T_maturity / device_dt);
        for(int i = 0; i < n_steps_T; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r_step_i, discount_factor_integral,
                              device_drift_table[i], G);
        }

        // Bumping r0 by ±ε shifts:
        //   r(T)            by  ±ε * e^{-aT}          (linearity of HW in r0)
        //   discount integral by  ±ε * B(0,T)           (integral of e^{-as} from 0 to T)
        float e_aT  = expf(-device_a * T_maturity);
        float B_0T  = BtT(0.0f, T_maturity, device_a);

        float rT_base  = r_step_i;
        float rT_up    = rT_base + epsilon * e_aT;
        float rT_down  = rT_base - epsilon * e_aT;

        float disc_base = expf(-discount_factor_integral);
        float disc_up   = disc_base * expf(-epsilon * B_0T);
        float disc_down = disc_base * expf(+epsilon * B_0T);

        // bond price at T for each scenario
        float P_base, P_up, P_down;
        if(P_market == nullptr){
            FlatCurve curve{device_a, device_sigma, device_r0};
            P_base = curve.P(T_maturity, S, rT_base);
            P_up   = curve.P(T_maturity, S, rT_up);
            P_down = curve.P(T_maturity, S, rT_down);
        } else {
            MarketCurve curve{device_a, device_sigma, P_market, f_market,
                              MAT_SPACING, N_MAT};
            P_base = curve.P(T_maturity, S, rT_base);
            P_up   = curve.P(T_maturity, S, rT_up);
            P_down = curve.P(T_maturity, S, rT_down);
        }

        float payoff_base = fmaxf(P_base - K, 0.0f);
        float payoff_up   = fmaxf(P_up   - K, 0.0f);
        float payoff_down = fmaxf(P_down - K, 0.0f);

        thread_zbc   = disc_base * payoff_base;

        // central difference — host divides by (2*eps) and (eps*eps)
        thread_delta = disc_up   * payoff_up - disc_down * payoff_down;
        thread_gamma = disc_up   * payoff_up
                     - 2.0f * disc_base * payoff_base
                     + disc_down * payoff_down;

        states[path_id] = local_state;
    }

    shared_zbc  [threadIdx.x] = thread_zbc;
    shared_delta[threadIdx.x] = thread_delta;
    shared_gamma[threadIdx.x] = thread_gamma;
    __syncthreads();

    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            shared_zbc  [threadIdx.x] += shared_zbc  [threadIdx.x + i];
            shared_delta[threadIdx.x] += shared_delta[threadIdx.x + i];
            shared_gamma[threadIdx.x] += shared_gamma[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(ZBC_estimator,   shared_zbc  [0]);
        atomicAdd(delta_estimator, shared_delta[0]);
        atomicAdd(gamma_estimator, shared_gamma[0]);
    }
}

void monteCarlo_delta_gamma(float T, float S, float K, curandState* d_states,
                             float* d_P_market = nullptr, float* d_f_market = nullptr,
                             float* h_P = nullptr, float* h_f = nullptr){

    float eps = 0.01f;

    float* d_ZBC   = nullptr;
    float* d_delta = nullptr;
    float* d_gamma = nullptr;
    cudaMalloc(&d_ZBC,   sizeof(float));
    cudaMalloc(&d_delta, sizeof(float));
    cudaMalloc(&d_gamma, sizeof(float));
    cudaMemset(d_ZBC,   0, sizeof(float));
    cudaMemset(d_delta, 0, sizeof(float));
    cudaMemset(d_gamma, 0, sizeof(float));

    mc_zbc_delta_gamma<<<NB, NTPB>>>(d_ZBC, d_delta, d_gamma, d_states,
                                      T, S, K, eps, d_P_market, d_f_market);
    cudaDeviceSynchronize();

    float h_ZBC, h_delta, h_gamma;
    cudaMemcpy(&h_ZBC,   d_ZBC,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_delta, d_delta, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gamma, d_gamma, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC   /= N_PATHS;
    h_delta /= N_PATHS * 2.0f * eps;
    h_gamma /= N_PATHS * eps * eps;

    auto ps = (d_P_market == nullptr)
        ? make_pricing_state(0.0f, T, S, K, host_r0, host_a, host_sigma,
                             FlatCurve{host_a, host_sigma, host_r0})
        : make_pricing_state(0.0f, T, S, K, host_r0, host_a, host_sigma,
                             MarketCurve{host_a, host_sigma, h_P, h_f, MAT_SPACING, N_MAT});

    float analytical_delta = delta_ZBC_from_state(ps);
    float analytical_gamma = gamma_ZBC_from_state(ps);

    const char* label = (d_P_market == nullptr) ? "" : " (market)";
    LOG_INFO("=== MC Delta/Gamma%s ===", label);
    LOG_INFO("MC Delta%s:      %.6f  |  Analytical: %.6f  |  Error: %.2e",
             label, h_delta, analytical_delta, fabsf(h_delta - analytical_delta));
    LOG_INFO("MC Gamma%s:      %.6f  |  Analytical: %.6f  |  Error: %.2e",
             label, h_gamma, analytical_gamma, fabsf(h_gamma - analytical_gamma));

    cudaFree(d_ZBC);
    cudaFree(d_delta);
    cudaFree(d_gamma);
}
#endif // MC_BOND_OPTIONS_CUH