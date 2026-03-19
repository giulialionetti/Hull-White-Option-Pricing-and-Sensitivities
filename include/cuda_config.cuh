#ifndef CUDA_CONFIG_CUH
#define CUDA_CONFIG_CUH

#define NTPB 1024
#define NB ((N_PATHS + NTPB - 1) / NTPB)

#endif // CUDA_CONFIG_CUH