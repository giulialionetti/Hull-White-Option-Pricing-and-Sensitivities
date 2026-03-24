#ifndef HW_OPTION_SENSITIVITIES_CUH
#define HW_OPTION_SENSITIVITIES_CUH

#include "hw_option_pricing.cuh"
#include "hw_bond_dsigma.cuh"


__host__ __device__ inline float vega_zbc(const EuroOption& o,
                                           float t, float T, float S,
                                           float a, float sigma){

    // probability density function of a Normal Gaussian                                       
    float phi_h    = expf(-o.h * o.h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    
    float PS_phi_h = o.P_S * phi_h; // P(t, S)*phi(h)

    float dsp_ds   = o.sigma_p / sigma;
    float B_S      = B(t, S, a);
    float B_T      = B(t, T, a);
    float dP_S_ds  = dP_dsigma(B_S, o.P_S, sigma, a, t);
    float dP_T_ds  = dP_dsigma(B_T, o.P_T, sigma, a, t);

    
// P(t,S)*phi(h)* dsigma_p/dsigma + N(h)*dP(t,S)dsigma - X*N(h-sigma_p)*dP(t,T)dsigma
    return PS_phi_h * dsp_ds + normcdff(o.h) * dP_S_ds - o.X * normcdff(o.h - o.sigma_p) * dP_T_ds;
}

__host__ __device__ inline float vega_zbp(const EuroOption& o,
                                           float t, float T, float S,
                                           float a, float sigma){

    float B_S      = B(t, S, a);
    float B_T      = B(t, T, a);
    float dP_S_ds  = dP_dsigma(B_S, o.P_S, sigma, a, t);
    float dP_T_ds  = dP_dsigma(B_T, o.P_T, sigma, a, t);

    // from Put-Call Parity ZBC-ZBP = P(t,S) - X* P(t,T)
    // dZBPdsigma = dZBCdsigma - d(P(t,S) + X* P(t,T))dsigma
    return vega_zbc(o, t, T, S, a, sigma) - dP_S_ds + o.X * dP_T_ds;

}

__host__ __device__ inline float volga_zbc(const EuroOption& o,
                                            float t, float T, float S,
                                            float a, float sigma){
    float B_S       = B(t, S, a);
    float B_T       = B(t, T, a);
    float dsp_ds    = o.sigma_p / sigma;
    float phi_h     = expf(-o.h * o.h * 0.5f) / sqrtf(2.0f * 3.14159265f);
    float phi_h_sp  = expf(-(o.h - o.sigma_p) * (o.h - o.sigma_p) * 0.5f)
                    / sqrtf(2.0f * 3.14159265f); // phi(h- sigma_p)

    float dP_S_ds   = dP_dsigma(B_S, o.P_S, sigma, a, t);
    float dP_T_ds   = dP_dsigma(B_T, o.P_T, sigma, a, t);
    float d2P_S_ds2 = d2P_dsigma2(B_S, o.P_S, sigma, a, t); // term 2
    float d2P_T_ds2 = d2P_dsigma2(B_T, o.P_T, sigma, a, t); // term3

    float srvn      = (1.0f - expf(-2.0f * a * t)) / (2.0f * a);
    - sigma * srvn * (B_S * B_S - B_T * B_T) / o.sigma_p;  

    float term1     = (dP_S_ds * phi_h - o.P_S * phi_h * o.h * dh_ds) * dsp_ds;
    float term2     = phi_h * dh_ds * dP_S_ds + normcdff(o.h) * d2P_S_ds2;
    float term3     = o.X * phi_h_sp * (dh_ds - dsp_ds) * dP_T_ds
                    + o.X * normcdff(o.h - o.sigma_p) * d2P_T_ds2;

    return term1 + term2 - term3;
}

__host__ __device__ inline float volga_zbp(const EuroOption& o,
                                            float t, float T, float S,
                                            float a, float sigma){
    float dP_S_ds   = dP_dsigma(B(t, S, a), o.P_S, sigma, a, t);
    float dP_T_ds   = dP_dsigma(B(t, T, a), o.P_T, sigma, a, t);
    float d2P_S_ds2 = d2P_dsigma2(B(t, S, a), o.P_S, sigma, a, t);
    float d2P_T_ds2 = d2P_dsigma2(B(t, T, a), o.P_T, sigma, a, t);

    return volga_zbc(o, t, T, S, a, sigma)
         - d2P_S_ds2
         + o.X * d2P_T_ds2;
}

#endif // HW_OPTION_SENSITIVITIES_CUH