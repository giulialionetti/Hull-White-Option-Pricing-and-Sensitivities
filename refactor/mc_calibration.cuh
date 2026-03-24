#ifndef MC_CALIBRATION_CUH
#define MC_CALIBRATION_CUH

#include "mc_engine.cuh"

__constant__ float device_drift_table[N_STEPS];
__constant__ float device_sensitivity_drift_table[N_STEPS];


inline void upload_drift(const float* th, float a, float sigma){
    float factor = (1.0f - expf(-a * host_dt)) / a;
    float host_drift_table[N_STEPS];
    float host_sensitivity_drift_table[N_STEPS];
    float host_sensitivity_drift_table2[N_STEPS];

    for(int i = 0; i < N_STEPS; i++){
        float s_mid     = (i + 0.5f) * host_dt;
        float t_idx     = s_mid / MAT_SPACING;
        int   idx       = (int)t_idx;
        float alpha     = t_idx - idx;
        float theta_mid = (idx >= N_MAT - 1)
            ? th[N_MAT - 1]
            : (1.0f - alpha) * th[idx] + alpha * th[idx + 1];

        host_drift_table[i] = theta_mid * factor;

        float s                  = i * host_dt;
        float s_plus_dt          = s + host_dt;
        float one_over_a_squared = 1.0f / (a * a);
        host_sensitivity_drift_table[i] = one_over_a_squared
                                        * 2.0f * sigma
                                        * expf(-a * s_plus_dt)
                                        * (coshf(a * s_plus_dt) - coshf(a * s));
    


    }

    cudaMemcpyToSymbol(device_drift_table,             host_drift_table,
                       N_STEPS * sizeof(float));
    cudaMemcpyToSymbol(device_sensitivity_drift_table, host_sensitivity_drift_table,
                       N_STEPS * sizeof(float));
}

inline void forward_rate(const float* log_P, float* f0){
    f0[0] = -(log_P[1] - log_P[0]) / MAT_SPACING;
    for(int i = 1; i < N_MAT - 1; i++)
        f0[i] = -(log_P[i+1] - log_P[i-1]) / (2.0f * MAT_SPACING);
    f0[N_MAT-1] = -(log_P[N_MAT-1] - log_P[N_MAT-2]) / MAT_SPACING;
}

inline void theta(const float* log_P, const float* f0,
                           float* th, float a, float sigma){
    for(int i = 1; i < N_MAT - 1; i++){
        float t_i = i * MAT_SPACING;
        float d2logP_dt2    = (log_P[i+1] - 2.0f*log_P[i] + log_P[i-1]) / (MAT_SPACING * MAT_SPACING);
        float convexity = (sigma * sigma / (2.0f * a)) * (1.0f - expf(-2.0f * a * t_i ));
        th[i]        = - d2logP_dt2 + a * f0[i] + convexity;
    }
    th[0]       = th[1];
    th[N_MAT-1] = th[N_MAT-2];
}

inline void calibrate(const float* h_P, float* f0, float a, float sigma){
    float log_P[N_MAT];

    for(int i = 0; i < N_MAT; i++)
        log_P[i] = logf(h_P[i]);

    forward_rate(log_P, f0);

    float th[N_MAT];
    theta(log_P, f0, th, a, sigma);

    upload_drift(th, a, sigma);
}

inline void init_drift(float a, float sigma, float r0){
    float th[N_MAT];
    for(int i = 0; i < N_MAT; i++){
        float t_i       = i * MAT_SPACING;
        float convexity = (sigma * sigma / (2.0f * a)) * (1.0f - expf(-2.0f * a * t_i));
        th[i]           = a * r0 + convexity;
    }
    upload_drift(th, a, sigma);
}

#endif // MC_CALIBRATION_CUH