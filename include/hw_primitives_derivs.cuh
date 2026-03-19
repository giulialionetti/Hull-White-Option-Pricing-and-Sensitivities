

#ifndef HW_PARAM_DERIVS_CUH
#define HW_PARAM_DERIVS_CUH

#include "hw_primitives.cuh"

// Derivatives of Hull-White model primitives w.r.t. model parameters a and σ.

struct BtT_a_derivs {
    float B;          // B(t,T)
    float dB_da;      // ∂B/∂a
    float d2B_da2;    // ∂²B/∂a²
};

__host__ __device__ inline BtT_a_derivs BtT_da(float t, float T, float a){
    float tau    = T - t;
    float e_atau = expf(-a * tau);
    float B      = (1.0f - e_atau) / a;

    BtT_a_derivs d;
    d.B       = B;
    d.dB_da   = -B / a + tau * e_atau / a;
    d.d2B_da2 = 2.0f * B / (a * a)
              - (tau * e_atau / a) * (2.0f / a + tau);
    return d;
}


struct srvn_a_derivs {
    float srvn;        // (1 - e^{-2at}) / (2a)
    float dsrvn_da;    // ∂srvn/∂a
    float d2srvn_da2;  // ∂²srvn/∂a²
};

__host__ __device__ inline srvn_a_derivs srvn_da(float t, float a){
    float e_2at = expf(-2.0f * a * t);
    float s     = (1.0f - e_2at) / (2.0f * a);

    srvn_a_derivs d;
    d.srvn       = s;
    d.dsrvn_da   = t * e_2at / a - s / a;
    d.d2srvn_da2 = 2.0f * s / (a * a)
                 - 2.0f * t * e_2at * (a * t + 1.0f) / (a * a);
    return d;
}


// We differentiate via ln A to avoid expanding a product of exponentials:
//
//   ln A = ln(P^M(0,T)/P^M(0,t)) + B·f^M(0,t) - (σ²/2)·srvn·B²
//
// The first term is constant w.r.t. a, so only the last two contribute.
// The curve-specific part (how f^M(0,t) is read) is encapsulated in
// dlnA_da / d2lnA_da2 methods added to each curve struct in hw_primitives.cuh.
//
//   ∂A/∂a   = A · ∂lnA/∂a
//   ∂²A/∂a² = A · [(∂lnA/∂a)² + ∂²lnA/∂a²]     (log-derivative identity)

struct AtT_a_derivs {
    float A;        // A(t,T)
    float dA_da;    // ∂A/∂a
    float d2A_da2;  // ∂²A/∂a²
};

template<typename Curve>
__host__ __device__ inline AtT_a_derivs AtT_da(float t, float T, float rt,
                                                float sigma, const Curve& curve){
    BtT_a_derivs  bda = BtT_da(t, T, curve.a);
    srvn_a_derivs sda = srvn_da(t, curve.a);

    float P = curve.P(t, T, rt);
    float A = P * expf(bda.B * rt);   // P = A·exp(-B·r)  →  A = P·exp(B·r)

    float lnA_1 = curve.dlnA_da (t, T, bda, sda);
    float lnA_2 = curve.d2lnA_da2(t, T, bda, sda);

    AtT_a_derivs d;
    d.A       = A;
    d.dA_da   = A * lnA_1;
    d.d2A_da2 = A * (lnA_1 * lnA_1 + lnA_2);
    return d;
}


//   ln P = ln A - B·r     (r constant w.r.t. a)
//
//   ∂ln P/∂a   = ∂ln A/∂a   - r · ∂B/∂a
//   ∂²ln P/∂a² = ∂²ln A/∂a² - r · ∂²B/∂a²
//
//   ∂P/∂a   = P · ∂ln P/∂a
//   ∂²P/∂a² = P · [(∂ln P/∂a)² + ∂²ln P/∂a²]
//
// Takes AtT_a_derivs by argument — AtT_da is always needed first anyway
// and recomputing it here would duplicate the curve.P() and expf() calls.

struct PtT_a_derivs {
    float P;
    float dP_da;
    float d2P_da2;
};

template<typename Curve>
__host__ __device__ inline PtT_a_derivs PtT_da(float t, float T, float rt,
                                                const AtT_a_derivs& ada,
                                                const BtT_a_derivs& bda){
    float lnP_1 = ada.dA_da   / ada.A - rt * bda.dB_da;
    float lnP_2 = ada.d2A_da2 / ada.A
                - (ada.dA_da  / ada.A) * (ada.dA_da / ada.A)
                - rt * bda.d2B_da2;

    PtT_a_derivs d;
    d.P       = ada.A * expf(-bda.B * rt);
    d.dP_da   = d.P * lnP_1;
    d.d2P_da2 = d.P * (lnP_1 * lnP_1 + lnP_2);
    return d;
}


#endif // HW_PARAM_DERIVS_CUH