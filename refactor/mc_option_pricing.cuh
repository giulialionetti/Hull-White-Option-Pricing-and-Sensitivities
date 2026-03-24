
#ifndef MC_OPTION_PRICING_CUH
#define MC_OPTION_PRICING_CUH


__global__ void simulate_option(float* out,
                                 curandState* states,
                                 const float* P0, const float* f0,
                                 float T, float S, float X,
                                 float a, float sigma, float r0){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_zbc[NTPB];
    __shared__ float s_zbp[NTPB];

    s_zbc[threadIdx.x] = 0.0f;
    s_zbp[threadIdx.x] = 0.0f;

    if(id < N_PATHS){
        curandState local_state = states[id];

        float r                 = r0;
        float discount_integral = 0.0f;

        int n_steps = (int)(T / device_dt);
        for(int i = 0; i < n_steps; i++){
            float G = curand_normal(&local_state);
            evolve_short_rate(r, discount_integral, device_drift_table[i], G);
        }

        float disc = expf(-discount_integral);
        float P_S  = P(P0, f0, T, S, r, a, sigma);

        s_zbc[threadIdx.x] = disc * fmaxf(P_S - X, 0.0f);
        s_zbp[threadIdx.x] = disc * fmaxf(X - P_S, 0.0f);

        states[id] = local_state;
    }

    __syncthreads();
    for(int i = NTPB/2; i > 0; i >>= 1){
        if(threadIdx.x < i){
            s_zbc[threadIdx.x] += s_zbc[threadIdx.x + i];
            s_zbp[threadIdx.x] += s_zbp[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(&out[0], s_zbc[0]);
        atomicAdd(&out[1], s_zbp[0]);
    }
}

#endif // MC_OPTION_PRICING_CUH