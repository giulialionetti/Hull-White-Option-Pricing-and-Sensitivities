#ifndef HW_GREEKS_SECOND_CUH
#define HW_GREEKS_SECOND_CUH

#include "hw_primitives.cuh"
#include "hw_pricing.cuh"
#include "hw_greeks_first.cuh"


// Section 1: Volga — ∂²/∂σ²
//
// Three channels contribute from differentiating the vega expression:
//
//   (A)  ∂(PS_phi_h · dsp_ds)/∂σ
//   (B)  ∂(-N(-h) · bS.dP_ds)/∂σ
//   (C)  ∂(K·N(σ_p-h) · bT.dP_ds)/∂σ
//
// In all three, the ∂h/∂σ terms do NOT cancel via PS_phi_h —
// unlike delta and vega, here they survive and produce the (α) term.
//

__host__ __device__ inline float volga_ZBC_from_state(const PricingState& ps){

    float dh_ds  = ps.dsp_ds * (ps.sigma_p - ps.h) / ps.sigma_p
                 - (ps.srvn / ps.dsp_ds) * (ps.bS.B * ps.bS.B - ps.bT.B * ps.bT.B);

    float phi_hm = expf(-(ps.h - ps.sigma_p) * (ps.h - ps.sigma_p) * 0.5f)
                 / sqrtf(2.0f * 3.14159265f);

    float alpha  = -ps.PS_phi_h * ps.h * dh_ds * ps.dsp_ds;
    float beta   =  ps.phi_h * (ps.dsp_ds + dh_ds) * ps.bS.dP_ds;
    float gamma  =  normcdff(ps.h) * ps.bS.d2P_ds2;
    float delta  = -ps.K * phi_hm * (dh_ds - ps.dsp_ds) * ps.bT.dP_ds;
    float eps    = -ps.K * normcdff(ps.h - ps.sigma_p) * ps.bT.d2P_ds2;

    return alpha + beta + gamma + delta + eps;
}

__host__ __device__ inline float volga_ZBP_from_state(const PricingState& ps){
    // parity: ∂²ZBC/∂σ² - ∂²ZBP/∂σ² = bS.d2P_ds2 - K·bT.d2P_ds2
    return volga_ZBC_from_state(ps)
         - ps.bS.d2P_ds2
         + ps.K * ps.bT.d2P_ds2;
}

// gamma ∂²/∂r²

__host__ __device__ inline float gamma_ZBC_from_state(const PricingState& ps){
    float BT_minus_BS = ps.bT.B - ps.bS.B;
    return  ps.bS.d2P_dr2 * normcdff( ps.h)
          - ps.K * ps.bT.d2P_dr2 * normcdff( ps.h - ps.sigma_p)
          + ps.PS_phi_h * (BT_minus_BS * BT_minus_BS) / ps.sigma_p;
}

__host__ __device__ inline float gamma_ZBP_from_state(const PricingState& ps){
    // parity: gamma_ZBC - gamma_ZBP = bS.d2P_dr2 - K*bT.d2P_dr2
    return gamma_ZBC_from_state(ps)
         - ps.bS.d2P_dr2
         + ps.K * ps.bT.d2P_dr2;
}


// Section 2: Templated entry points

template<typename Curve>
__host__ __device__ inline float volga_ZBC_impl(float t, float T, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    return volga_ZBC_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

template<typename Curve>
__host__ __device__ inline float volga_ZBP_impl(float t, float T, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    return volga_ZBP_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

// Backward-compatible flat-curve wrappers
__host__ __device__ inline float volga_ZBC(float t, float T, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return volga_ZBC_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float volga_ZBP(float t, float T, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return volga_ZBP_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

template<typename Curve>
__host__ __device__ inline float gamma_ZBC_impl(float t, float T, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    return gamma_ZBC_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

template<typename Curve>
__host__ __device__ inline float gamma_ZBP_impl(float t, float T, float S, float K,
                                                 float rt, float a, float sigma,
                                                 const Curve& curve){
    return gamma_ZBP_from_state(make_pricing_state(t, T, S, K, rt, a, sigma, curve));
}

// Backward-compatible flat-curve wrappers
__host__ __device__ inline float gamma_ZBC(float t, float T, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return gamma_ZBC_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

__host__ __device__ inline float gamma_ZBP(float t, float T, float S, float K,
                                            float rt, float a, float sigma, float r0){
    return gamma_ZBP_impl(t, T, S, K, rt, a, sigma, FlatCurve{a, sigma, r0});
}

#endif // HW_GREEKS_SECOND_CUH