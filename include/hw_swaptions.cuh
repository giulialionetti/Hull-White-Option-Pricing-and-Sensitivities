#ifndef SWAPTIONS_CUH
#define SWAPTIONS_CUH

#include "hw_model.cuh"
#include "hw_option_pricing.cuh"
#include "hw_option_sensitivities.cuh"

__host__ __device__ inline float par_swap_rate(float T, const float* tenor_dates,
                                                int n_tenors, const float* P0){
    float P_T  = interpolate(P0, T);
    float P_Tn = interpolate(P0, tenor_dates[n_tenors - 1]);

    float annuity = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float delta_i = (i == 0) ? tenor_dates[0] - T
                                 : tenor_dates[i] - tenor_dates[i-1];
        annuity += delta_i * interpolate(P0, tenor_dates[i]);
    }
    return (P_T - P_Tn) / annuity;
}

__host__ __device__ inline float swap_value_at_r(float r, float T,
                                                   const float* tenor_dates, int n_tenors,
                                                   const float* c,
                                                   const float* P0, const float* f0,
                                                   float a, float sigma){
    float val = 0.0f;
    for(int i = 0; i < n_tenors; i++)
        val += c[i] * P(P0, f0, T, tenor_dates[i], r, a, sigma);
    return val;
}

inline float critical_rate_r_star(float T, const float* tenor_dates, int n_tenors,
                                   const float* c,
                                   const float* P0, const float* f0,
                                   float a, float sigma){
    float lo = -0.5f, hi = 0.5f;
    for(int iter = 0; iter < 100; iter++){
        float mid = 0.5f * (lo + hi);
        float val = swap_value_at_r(mid, T, tenor_dates, n_tenors, c, P0, f0, a, sigma);
        if(val > 1.0f) lo = mid;
        else           hi = mid;
    }
    return 0.5f * (lo + hi);
}

inline float analytical_swaption(float T, const float* tenor_dates, int n_tenors,
                                  const float* c,
                                  const float* P0, const float* f0,
                                  float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);

    float price = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        price += c[i] * ZBP(o);
    }
    return price;
}

inline float analytical_swaption_vega(float T, const float* tenor_dates, int n_tenors,
                                       const float* c,
                                       const float* P0, const float* f0,
                                       float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);

    float vega = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i    = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        float B_T_Ti = B(T, tenor_dates[i], a);

        float dXi_dsigma = -X_i * (sigma / (2.0f * a))
                          * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti;

        EuroOption o    = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        float P_0T      = o.P_T;
        float sigma_p   = o.sigma_p;
        float h         = o.h;
        float dZBP_dXi  = P_0T * normcdff(-h + sigma_p);

        vega += c[i] * (vega_zbp(o, 0.0f, T, tenor_dates[i], a, sigma)
                      + dZBP_dXi * dXi_dsigma);
    }
    return vega;
}

inline float analytical_swaption_volga(float T, const float* tenor_dates, int n_tenors,
                                        const float* c,
                                        const float* P0, const float* f0,
                                        float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);

    float volga = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i    = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        float B_T_Ti = B(T, tenor_dates[i], a);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);

        float dXi_dsigma = -X_i * (sigma / (2.0f * a))
                          * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti;

        float P_0T      = o.P_T;
        float phi_h_sp  = expf(-(o.h - o.sigma_p) * (o.h - o.sigma_p) * 0.5f)
                        / sqrtf(2.0f * 3.14159265f);
        float dZBP_dXi   = P_0T * normcdff(-o.h + o.sigma_p);
        float d2ZBP_dXi2 = P_0T * phi_h_sp / (X_i * o.sigma_p);

        volga += c[i] * (volga_zbp(o, 0.0f, T, tenor_dates[i], a, sigma)
                       + 2.0f * dZBP_dXi * dXi_dsigma
                       + d2ZBP_dXi2 * dXi_dsigma * dXi_dsigma);
    }
    return volga;
}

inline float analytical_swaption_delta(float T, const float* tenor_dates, int n_tenors,
                                        const float* c,
                                        const float* P0, const float* f0,
                                        float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);

    float delta = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        delta += c[i] * delta_zbp(o, 0.0f, T, tenor_dates[i], a);
    }
    return delta;
}

inline float analytical_swaption_gamma(float T, const float* tenor_dates, int n_tenors,
                                        const float* c,
                                        const float* P0, const float* f0,
                                        float a, float sigma, float r0){
    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c, P0, f0, a, sigma);

    float gamma = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = P(P0, f0, T, tenor_dates[i], r_star, a, sigma);
        EuroOption o = euro_option(P0, f0, 0.0f, T, tenor_dates[i], X_i, r0, a, sigma);
        gamma += c[i] * gamma_zbp(o, 0.0f, T, tenor_dates[i], a);
    }
    return gamma;
}

#endif // SWAPTIONS_CUH