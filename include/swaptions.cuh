
#ifndef SWAPTIONS_CUH
#define SWAPTIONS_CUH

#include "mc.cuh"

// at the money K = par swap rate
__host__ __device__ inline float par_swap_rate(float T, const float* tenor_dates, int n_tenors, const float* host_Price,
     float maturity_spacing, int n_maturities){

      float Bond_price_at_expiry_T = interpolate(host_Price,
             T, maturity_spacing, n_maturities);
      float Bond_price_at_tenor_Tn = interpolate(host_Price,
             tenor_dates[n_tenors -1], maturity_spacing, n_maturities);

      float annuity = 0.0f;
       

     for(int i=0; i< n_tenors; i++){
          float delta_i = (i == 0) ? tenor_dates[0] - T
                                 : tenor_dates[i] - tenor_dates[i-1];
        annuity += delta_i * interpolate(host_Price, tenor_dates[i], maturity_spacing, n_maturities);
      }

      return (Bond_price_at_expiry_T - Bond_price_at_tenor_Tn) / annuity;
}

__host__ __device__ inline float swap_value_at_r(float r, float T, 
const float* tenor_dates, const int n_tenors, const float* c, const float* host_Price,
const float* host_fwd_rate, float a, float sigma, float maturity_spacing, int n_maturities){

    MarketCurve curve{a, sigma, host_Price, host_fwd_rate, maturity_spacing, n_maturities};
    float swap_value_r = 0.0f;

    for(int i=0; i < n_tenors; i++){
        swap_value_r += c[i] * curve.P(T, tenor_dates[i], r);
    }

    return swap_value_r;
}

inline float critical_rate_r_star(float T, const float* tenor_dates, int n_tenors,
                                   const float* c, const float* host_Price,
                                   const float* host_fwd_rate, float a, float sigma,
                                   float maturity_spacing, int n_maturities){
    float lo = -0.5f, hi = 0.5f;
    for(int iter = 0; iter < 100; iter++){
        float mid = 0.5f * (lo + hi);
        float val = swap_value_at_r(mid, T, tenor_dates, n_tenors, c,
                                    host_Price, host_fwd_rate, a, sigma,
                                    maturity_spacing, n_maturities);
        if(val > 1.0f) lo = mid;
        else           hi = mid;
    }
    return 0.5f * (lo + hi);
}

inline float analytical_swaption(float T, const float* tenor_dates, int n_tenors,
                                  const float* c, const float* host_Price,
                                  const float* host_fwd_rate, float a, float sigma,
                                  float r0, float maturity_spacing, int n_maturities){

    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c,
                                        host_Price, host_fwd_rate, a, sigma,
                                        maturity_spacing, n_maturities);

    MarketCurve curve{a, sigma, host_Price, host_fwd_rate, maturity_spacing, n_maturities};

    float swaption_price = 0.0f;
    for(int i = 0; i < n_tenors; i++){
        float X_i = curve.P(T, tenor_dates[i], r_star);
        swaption_price += c[i] * ZBP_from_state(
    make_pricing_state(0.0f, T, tenor_dates[i], X_i, r0, a, sigma, curve));
    }
    return swaption_price;
}

// once with respect to sigma

inline float analytical_payer_swaption_vega(float T, const float* tenor_dates, int n_tenors,
                                       const float* c, const float* host_Price,
                                       const float* host_fwd_rate, float a, float sigma,
                                       float r0, float maturity_spacing, int n_maturities){

    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c,
                                        host_Price, host_fwd_rate, a, sigma,
                                        maturity_spacing, n_maturities);

    MarketCurve curve{a, sigma, host_Price, host_fwd_rate, maturity_spacing, n_maturities};

    float vega = 0.0f;
    
    for(int i = 0; i < n_tenors; i++){
    float X_i    = curve.P(T, tenor_dates[i], r_star);
    float B_T_Ti = BtT(T, tenor_dates[i], a);
    
    // dX_i/dsigma from convexity term in A(T,Ti)
    // where: A(T,ti) = (P_market(0, ti) / P_market(0, T)) *
    //  exp (B(T, ti)* f_market(0, T) - (sigma)^2/4a(1-exp(-2aT)))* B(T, ti)^2
    // where: P_market, f_market are known, B(T, ti) does not depend on sigma 
    // where: Xi = A(T, ti)*exp(-B(T, ti) r*) with r* recovered from market data 
    // and therefore not explicitly dependent on sigma 
    float dXi_dsigma = - X_i * (sigma / (2.0f * a))
                      * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti;

    // ∂ZBP/∂X_i = - P(0,T) * N(-h + sigma_p)  
    float bond_0_T  = curve.P(0.0f, T, r0);
    float bond_0_Ti = curve.P(0.0f, tenor_dates[i], r0);
    float B_T_Ti_s  = BtT(T, tenor_dates[i], a);
    float sigma_p   = sigma * sqrtf((1.0f - expf(-2.0f * a * T)) / (2.0f * a)) * B_T_Ti_s;
    float h         = (1.0f / sigma_p) * logf(bond_0_Ti / (bond_0_T * X_i)) + sigma_p / 2.0f;
    float dZBP_dXi  = bond_0_T * normcdff(-h + sigma_p);

    vega += c[i] * (vega_ZBP_impl(0.0f, T, tenor_dates[i], X_i, r0, a, sigma, curve)
                  + dZBP_dXi * dXi_dsigma);
}
    return vega;
}


inline float analytical_payer_swaption_volga(float T, const float* tenor_dates, int n_tenors,
                                              const float* c, const float* host_Price,
                                              const float* host_fwd_rate, float a, float sigma,
                                              float r0, float maturity_spacing, int n_maturities){

    float r_star = critical_rate_r_star(T, tenor_dates, n_tenors, c,
                                        host_Price, host_fwd_rate, a, sigma,
                                        maturity_spacing, n_maturities);

    MarketCurve curve{a, sigma, host_Price, host_fwd_rate, maturity_spacing, n_maturities};
   

    float volga = 0.0f;

    for(int i = 0; i < n_tenors; i++){
        float X_i    = curve.P(T, tenor_dates[i], r_star);
        float B_T_Ti = BtT(T, tenor_dates[i], a);

        // same as in VEGA
        float dXi_dsigma = - X_i * (sigma / (2.0f * a))
                      * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti;

        // ∂²Xi/∂σ² 
        float d2Xi_ds2    = X_i * (sigma / (2.0f * a))
                      * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti
                          * (sigma * sigma * (sigma / (2.0f * a))
                      * (1.0f - expf(-2.0f * a * T)) * B_T_Ti * B_T_Ti - 1.0f);

        
        PricingState ps   = make_pricing_state(0.0f, T, tenor_dates[i], X_i,
                                               r0, a, sigma, curve);
        float phi_hm      = expf(-(ps.h - ps.sigma_p) * (ps.h - ps.sigma_p) * 0.5f)
                          / sqrtf(2.0f * 3.14159265f);   // φ(h - σ_p)

        // ∂ZBP/∂Xi = P(0,T)·Φ(-h+σ_p)
        float dZBP_dXi    = ps.bT.P * normcdff(-ps.h + ps.sigma_p);

        //  ∂²ZBP/∂Xi² = P(0,T)·φ(-h+σ_p)/(Xi·σ_p) 
        float d2ZBP_dXi2  = ps.bT.P * phi_hm / (X_i * ps.sigma_p);

        // cross ∂²ZBP/∂σ∂Xi 
        // from differentiating dZBP_dXi w.r.t σ:
        // ∂h/∂σ = -h·dsp_ds/σ_p  (from h = lnM/σ_p + σ_p/2, dsp_ds = σ_p/σ)
        float dh_ds       = -ps.h * ps.dsp_ds / ps.sigma_p;
        float d2ZBP_dsdXi = ps.bT.P * phi_hm * (ps.dsp_ds - dh_ds) / (X_i * ps.sigma_p);

         
        float zbp_volga   = volga_ZBP_volga_from_state(ps);   

        volga += c[i] * (zbp_volga
                       + 2.0f * d2ZBP_dsdXi * dXi_ds
                       + d2ZBP_dXi2         * dXi_ds * dXi_ds
                       + dZBP_dXi           * d2Xi_ds2);
    }
    return volga;
}















#endif // SWAPTIONS_CUH