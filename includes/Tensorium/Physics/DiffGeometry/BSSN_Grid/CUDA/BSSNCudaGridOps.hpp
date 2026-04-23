#pragma once

#include "../Fields/BSSNGridViews.hpp"

#include <cstddef>
#include <stdexcept>

namespace tensorium_RG::bssn::cuda {

template <typename T> struct GaugeRHSCudaConfig {
    T ko_sigma = T(0);
    T ko_boundary_floor = T(0);
    T beta_B_coeff = T(0.75);
    T eta_coeff = T(2);
    T lapse_harmonicf = T(1);
    T lapse_harmonic = T(0);
    T lapse_oplog = T(2);
    T lapse_advect = T(1);
    T shift_advect = T(1);
    T slow_start_lapse_factor = T(1);
    int spatial_order = 6;
    std::size_t ko_boundary_width = 0;
    bool use_theta_in_lapse = true;
    bool use_shift_advection = true;
};

template <typename T> struct ScalarRHSCudaConfig {
    T ko_sigma = T(0);
    T ko_boundary_floor = T(0);
    T kappa1 = T(0);
    T kappa2 = T(0);
    T chi_div_floor = T(-1000);
    int spatial_order = 6;
    std::size_t ko_boundary_width = 0;
    bool covariant_z4 = true;
};

template <typename T> struct RHSSommerfeldCudaConfig {
    std::size_t collar_width = 1;
    bool face_enabled[3][2] = {{false, false}, {false, false}, {false, false}};
    T alpha_speed = T(1);
    T chi_speed = T(1);
    T K_speed = T(1);
    T Theta_speed = T(1);
    T beta_speed = T(1);
    T B_speed = T(1);
    T tildeGamma_speed = T(1);
    T Z_speed = T(1);
    T gamma_tilde_speed = T(1);
    T A_tilde_speed = T(1);
};

template <typename T> struct RHSSpongeCudaConfig {
    bool enabled = false;
    std::size_t width = 0;
    T strength = T(0);
    T exponent = T(2);
    bool active_face[3][2] = {{true, true}, {true, true}, {true, true}};
    bool reflective_face[3][2] = {{false, false}, {false, false}, {false, false}};
};

#ifdef TENSORIUM_CUDA

template <typename T>
void copy_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, std::size_t padding);

template <typename T>
void copy_stage_reference(BSSNGridView<const T> src, BSSNRHSWorkspaceView<T> dst,
                          std::size_t padding);

template <typename T>
void accumulate_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, T delta,
                                std::size_t padding);

template <typename T>
void accumulate_stage_reference(BSSNGridView<const T> src, BSSNRHSWorkspaceView<T> dst, T delta,
                                std::size_t padding);

template <typename T>
void compute_gauge_rhs(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                       GaugeRHSCudaConfig<T> config, std::size_t padding);

template <typename T>
void compute_scalar_rhs(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                        Field3DView<const T> z4_conformal_trace, ScalarRHSCudaConfig<T> config,
                        std::size_t padding);

template <typename T>
void apply_rhs_sommerfeld(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                          RHSSommerfeldCudaConfig<T> config);

template <typename T>
void apply_rhs_sponge(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                      RHSSpongeCudaConfig<T> config);

template <typename T>
void stabilize_nonfinite_rhs_collar(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                                    std::size_t collar_width);

template <typename T>
void recompose_rhs_K_from_khat(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                               std::size_t padding);

template <typename T> void enforce_algebraic_constraints(BSSNGridView<T> grid);

template <typename T>
void apply_explicit_stage_update(BSSNGridView<T> grid, BSSNGridView<const T> stage,
                                 BSSNRHSWorkspaceView<const T> rhs, T gam0, T gam1, T beta_dt,
                                 std::size_t padding);

template <typename T>
void apply_explicit_stage_update(BSSNGridView<T> grid, BSSNRHSWorkspaceView<const T> stage,
                                 BSSNRHSWorkspaceView<const T> rhs, T gam0, T gam1, T beta_dt,
                                 std::size_t padding);

template <typename T> void apply_alpha_floor(BSSNGridView<T> grid, T floor);

template <typename T> void apply_chi_floor(BSSNGridView<T> grid, T floor);

#else

template <typename T>
inline void copy_stage_reference(BSSNGridView<const T>, BSSNGridView<T>, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void copy_stage_reference(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void accumulate_stage_reference(BSSNGridView<const T>, BSSNGridView<T>, T, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void accumulate_stage_reference(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>, T,
                                       std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void compute_gauge_rhs(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>,
                              GaugeRHSCudaConfig<T>, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void compute_scalar_rhs(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>,
                               Field3DView<const T>, ScalarRHSCudaConfig<T>, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void apply_rhs_sommerfeld(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>,
                                 RHSSommerfeldCudaConfig<T>) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void apply_rhs_sponge(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>,
                             RHSSpongeCudaConfig<T>) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void stabilize_nonfinite_rhs_collar(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>,
                                           std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void recompose_rhs_K_from_khat(BSSNGridView<const T>, BSSNRHSWorkspaceView<T>,
                                      std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T> inline void enforce_algebraic_constraints(BSSNGridView<T>) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void apply_explicit_stage_update(BSSNGridView<T>, BSSNGridView<const T>,
                                        BSSNRHSWorkspaceView<const T>, T, T, T, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void apply_explicit_stage_update(BSSNGridView<T>, BSSNRHSWorkspaceView<const T>,
                                        BSSNRHSWorkspaceView<const T>, T, T, T, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T> inline void apply_alpha_floor(BSSNGridView<T>, T) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T> inline void apply_chi_floor(BSSNGridView<T>, T) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

#endif

} // namespace tensorium_RG::bssn::cuda
