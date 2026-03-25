#ifndef HW_BOND_DERIV_A_CUH
#define HW_BOND_DERIV_A_CUH

#include "hw_model.cuh"

__host__ __device__ inline float dB_da(float t, float T, float a){
    float time_to_maturity  = T - t;
    float b    = B(t, T, a);
    float e_at = expf(-a * time_to_maturity);
    return (-b + time_to_maturity * e_at) / a;
}

__host__ __device__ inline float d2B_da2(float t, float T, float a){
    float time_to_maturity = T - t;
    float e_at = expf(-a * time_to_maturity);
    float a_squared = (a * a);
    float a_cube = (a * a * a);
    return 2.0f / (a_cube)  - (2.0f / (a_cube) + 2.0f * time_to_maturity / (a_squared) 
    + time_to_maturity * time_to_maturity / a) * e_at;
}

__host__ __device__ inline float dlnA_da(float t, float T, float a, float sigma,
                                          float f0t, float B, float dB){
    float srvn       = (1.0f - expf(-2.0f * a * t)) / (2.0f * a);
    float dsrvn      = -srvn / a + t * expf(-2.0f * a * t) / a;
    float hs2        = 0.5f * sigma * sigma;
    return f0t * dB - hs2 * (dsrvn * B * B + 2.0f * srvn * B * dB);
}

__host__ __device__ inline float d2lnA_da2(float t, float T, float a, float sigma,
                                             float f0t, float B, float dB, float d2B){
    float e_2at      = expf(-2.0f * a * t);
    float srvn       = (1.0f - e_2at) / (2.0f * a);
    float dsrvn      = -srvn / a + t * e_2at / a;
    float d2srvn     = 2.0f * srvn / (a * a)
                     - 2.0f * t * e_2at * (a * t + 1.0f) / (a * a);
    float hs2        = 0.5f * sigma * sigma;
    return f0t * d2B
         - hs2 * (d2srvn * B  * B
                + 4.0f * dsrvn * B  * dB
                + 2.0f * srvn  * dB * dB
                + 2.0f * srvn  * B  * d2B);
}

__host__ __device__ inline float dP_da(float P, float rt,
                                        float dB, float dlnA){
    return P * (dlnA - rt * dB);
}

__host__ __device__ inline float d2P_da2(float P, float rt,
                                          float dB, float d2B,
                                          float dlnA, float d2lnA){
    float dlnP  = dlnA - rt * dB;
    float d2lnP = d2lnA - rt * d2B;
    return P * (dlnP * dlnP + d2lnP);
}

#endif // HW_BOND_DERIV_A_CUH