#pragma once

#include <algorithm>

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"
#include "BSSNEvolutionCommon.hpp"

namespace tensorium_RG::bssn {

template <typename T> struct GaugeParameters {
    T beta_B_coeff = T(3.0 / 4.0);
    T eta = T(1);
};

template <typename T>
inline void compute_rhs_alpha(const BSSNGridSoA<T> &G, Field3D<T> &rhs_alpha,
                              size_t padding = 4) {
    using namespace tensorium_RG::fd;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total = G.alpha.st.nx_tot * G.alpha.st.ny_tot * G.alpha.st.nz_tot;
    std::fill(rhs_alpha.ptr(), rhs_alpha.ptr() + total, T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);
                const T alpha = G.alpha.ptr()[id];
                const T K = G.K.ptr()[id];
                const T beta_x = G.beta[0].ptr()[id];
                const T beta_y = G.beta[1].ptr()[id];
                const T beta_z = G.beta[2].ptr()[id];
                const T advection = beta_x * T(Dx(G.alpha, i, j, k, G.dx)) +
                                    beta_y * T(Dy(G.alpha, i, j, k, G.dy)) +
                                    beta_z * T(Dz(G.alpha, i, j, k, G.dz));
                rhs_alpha.ptr()[id] = advection - T(2) * alpha * K;
            }
}

template <typename T>
inline void compute_rhs_beta(const BSSNGridSoA<T> &G, Field3D<T> rhs[3],
                             const GaugeParameters<T> &params = {}, size_t padding = 4) {
    using namespace tensorium_RG::fd;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total = G.beta[0].st.nx_tot * G.beta[0].st.ny_tot * G.beta[0].st.nz_tot;
    for (int c = 0; c < 3; ++c)
        std::fill(rhs[c].ptr(), rhs[c].ptr() + total, T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.beta[0].idx(i, j, k);
                const T      beta_vec[3] = {G.beta[0].ptr()[id], G.beta[1].ptr()[id],
                                            G.beta[2].ptr()[id]};
                for (int comp = 0; comp < 3; ++comp) {
                    const Field3D<T> &field = G.beta[comp];
                    const T adv = beta_vec[0] * T(Dx(field, i, j, k, G.dx)) +
                                  beta_vec[1] * T(Dy(field, i, j, k, G.dy)) +
                                  beta_vec[2] * T(Dz(field, i, j, k, G.dz));
                    rhs[comp].ptr()[id] = adv + params.beta_B_coeff * G.B[comp].ptr()[id];
                }
            }
}

template <typename T>
inline void compute_rhs_B(const BSSNGridSoA<T> &G, const Field3D<T> rhs_Gamma[3],
                          Field3D<T> rhs_B[3], const GaugeParameters<T> &params = {},
                          size_t padding = 4) {
    using namespace tensorium_RG::fd;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total = G.B[0].st.nx_tot * G.B[0].st.ny_tot * G.B[0].st.nz_tot;
    for (int c = 0; c < 3; ++c)
        std::fill(rhs_B[c].ptr(), rhs_B[c].ptr() + total, T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.B[0].idx(i, j, k);
                const T      beta_vec[3] = {G.beta[0].ptr()[id], G.beta[1].ptr()[id],
                                            G.beta[2].ptr()[id]};
                for (int comp = 0; comp < 3; ++comp) {
                    const Field3D<T> &field = G.B[comp];
                    const T adv = beta_vec[0] * T(Dx(field, i, j, k, G.dx)) +
                                  beta_vec[1] * T(Dy(field, i, j, k, G.dy)) +
                                  beta_vec[2] * T(Dz(field, i, j, k, G.dz));
                    rhs_B[comp].ptr()[id] = adv + rhs_Gamma[comp].ptr()[id] -
                                             params.eta * G.B[comp].ptr()[id];
                }
            }
}

} // namespace tensorium_RG::bssn
