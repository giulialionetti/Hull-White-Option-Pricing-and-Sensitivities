#ifndef HW_OPTION_PRICING_CUH
#define HW_OPTION_PRICING_CUH

#include "hw_model.cuh"

struct EuroOption {
    float P_S; // P(t, S)
    float P_T; // P(t, T)
    float sigma_p;
    float h;
    float X; // Strike 
};

__host__ __device__ inline EuroOption euro_option(const float* P0, const float* f0,
                                                    float t, float T, float S, float X,
                                                    float rt, float a, float sigma){
    float P_S     = P(P0, f0, t, S, rt, a, sigma);
    float P_T     = P(P0, f0, t, T, rt, a, sigma);
    float B_TS    = B(T, S, a);
    float sigma_p = sigma * sqrtf((1.0f - expf(-2.0f * a * (T - t))) / (2.0f * a)) * B_TS;
    float h       = (1.0f / sigma_p) * logf(P_S / (P_T * X)) + sigma_p / 2.0f;

    return {P_S, P_T, sigma_p, h, X};
}

// European Call Option price 
__host__ __device__ inline float ZBC(const EuroOption& o){
    // P(t, S)*normcdf(h) - X*P(t,T)*normcdf(h-sigma_p)
    return o.P_S * normcdff(o.h) - o.X * o.P_T * normcdff(o.h - o.sigma_p);
}

// European Put Option Price
__host__ __device__ inline float ZBP(const EuroOption& o){
    // X*P(t, T)normcdf(-h+sigma_p)- P(t,S)*normcdf(-h)
    return o.X * o.P_T * normcdff(-o.h + o.sigma_p) - o.P_S * normcdff(-o.h);
}

#endif // HW_OPTION_PRICING_CUH