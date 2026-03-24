#ifndef HW_CONSTANTS_CUH
#define HW_CONSTANTS_CUH

#define N_PATHS     (1024 * 1024)
#define N_STEPS     1000
#define T_FINAL     10.0f
#define N_MAT       100
#define MAT_SPACING 0.1f
#define SAVE_STRIDE (N_STEPS / N_MAT)
#define MAX_TENORS  10
#define NTPB        1024
#define NB          ((N_PATHS + NTPB - 1) / NTPB)

const float host_dt = T_FINAL / N_STEPS;

#endif // HW_CONSTANTS_CUH