#ifndef HW_BOND_DSIGMA_CUH
#define HW_BOND_DSIGMA_CUH


#include "hw_model.cuh"

__host__ __device__ inline float dP_dsigma(float B, float P, float sigma, float a, float t){
    float B_squared =  B * B ;
    float factor = - (sigma * (1.0f - expf(-2.0f*a*t)))/ (2.0f*a);
    
    return P * factor * B_squared;
}

__host__ __device__ inline float d2P_dsigma2(float B, float P, float sigma, float a, float t){
    
    float factor = - (sigma * (1.0f - expf(-2.0f*a*t)))/ (2.0f*a);
    float B_squared =  B * B ;
    float dPdsigma = P * factor * B_squared ;
    float dfactordsigma = factor / sigma ;


    float term1 = dPdsigma * factor * B_squared ;
    float term2 = P * dfactordsigma * B_squared;

    return term1 + term2;
}



#endif // HW_BOND_DSIGMA_CUH