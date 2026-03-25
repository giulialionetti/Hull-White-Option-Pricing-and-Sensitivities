#ifndef HW_MODEL_CUH
#define HW_MODEL_CUH

#include <cmath>
#include "mc_engine.cuh"
#include "hw_constants.cuh"

__host__ __device__ inline float B(float t, float T, float a){
    return (1.0f - expf(-a * (T - t))) / a;
}

__host__ __device__ inline float A(float b, float t, float a, float sigma,
                                    float P0T, float P0t, float f0t){
    
    float P0T_P0t = P0T / P0t;  
    float sigma_squared = sigma * sigma;
    float short_rate_var_normalized = sigma_squared * (1.0f - expf(-2.0f * a * t)) / (4.0f * a);
    float B_squared = b * b;                
    return P0T_P0t * expf(b * f0t - short_rate_var_normalized * B_squared);
}

__host__ __device__ inline float interpolate(const float* data, float t){
    int   idx   = (int)(t / MAT_SPACING);
    if(idx >= N_MAT - 1) return data[N_MAT - 1];
    float alpha = t / MAT_SPACING - idx;
    return data[idx] * (1.0f - alpha) + data[idx + 1] * alpha;
}

__host__ __device__ inline float P(const float* P0, const float* f0,
                                    float t, float T, float rt,
                                    float a, float sigma){
    float b   = B(t, T, a);
    float P0T = interpolate(P0, T);
    float P0t = (t == 0.0f) ? 1.0f : interpolate(P0, t);
    float f0t = interpolate(f0, t);
    return A(b, t, a, sigma, P0T, P0t, f0t) * expf(-b * rt);
}

#endif // HW_MODEL_CUH