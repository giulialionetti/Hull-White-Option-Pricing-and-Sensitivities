#ifndef HW_BOND_DR_CUH
#define HW_BOND_DR_CUH

#include "hw_model.cuh"

__host__ __device__ inline float dP_dr(float B, float P){
    return -B * P;
}

__host__ __device__ inline float d2P_dr2(float B, float P){
    return B * B * P;
}

#endif // HW_BOND_DR_CUH