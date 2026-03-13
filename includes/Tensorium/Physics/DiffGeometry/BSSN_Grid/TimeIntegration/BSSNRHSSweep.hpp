#pragma once

#include "../Evolution/BSSNEvolutionATilde.hpp"
#include "../Evolution/BSSNEvolutionChi.hpp"
#include "../Evolution/BSSNEvolutionGamma.hpp"
#include "../Evolution/BSSNEvolutionGammaTilde.hpp"
#include "../Evolution/BSSNEvolutionGauge.hpp"
#include "../Evolution/BSSNEvolutionK.hpp"
#include "../Evolution/BSSNEvolutionZ4C.hpp"

namespace tensorium_RG::bssn {

template <typename T>
inline void evaluate_rhs_sweep_core(const BSSNGridSoA<T> &grid, Field3D<T> &rhs_alpha,
                                    Field3D<T> &rhs_chi, Field3D<T> &rhs_K,
                                    Field3D<T> &rhs_Theta, Field3D<T> rhs_beta[3],
                                    Field3D<T> rhs_B[3], Field3D<T> rhs_gamma_tilde[6],
                                    Field3D<T> rhs_A_tilde[6], Field3D<T> rhs_tildeGamma[3],
                                    Field3D<T> rhs_Z[3], const GaugeParameters<T> &params,
                                    size_t padding, Field3D<T> *z4_conformal_trace_cache) {
#pragma omp parallel
    {
        set_rhs_kernel_team_mode(true);
        compute_rhs_Gamma(grid, rhs_tildeGamma, grid.Z, grid.Theta, params, padding);
        compute_rhs_B(grid, rhs_tildeGamma, rhs_B, params, padding);
        compute_rhs_beta(grid, rhs_beta, params, padding);
        compute_rhs_alpha(grid, rhs_alpha, padding, params);
        compute_rhs_chi(grid, rhs_chi, padding, params);
        compute_rhs_gamma_tilde(grid, rhs_gamma_tilde, padding, params);
        compute_rhs_A_tilde(grid, rhs_A_tilde, padding, params, z4_conformal_trace_cache);
        compute_rhs_K(grid, rhs_K, padding, params);
        compute_rhs_Theta(grid, rhs_Theta, params, padding, z4_conformal_trace_cache);
        compute_rhs_Z(grid, rhs_Z, params, padding);
        set_rhs_kernel_team_mode(false);
    }
}

template <typename T>
inline void recompose_rhs_K_from_khat_core(const BSSNGridSoA<T> &grid, Field3D<T> &rhs_K,
                                           const Field3D<T> &rhs_Theta, size_t padding) {
    for_each_interior_index_parallel(grid, padding, [&](size_t, size_t, size_t, size_t idx) {
        rhs_K.ptr()[idx] += T(2) * rhs_Theta.ptr()[idx];
    });
}

} // namespace tensorium_RG::bssn
