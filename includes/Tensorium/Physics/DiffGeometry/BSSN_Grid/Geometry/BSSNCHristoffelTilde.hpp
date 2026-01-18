#pragma once
#include "BSSNGamma.hpp"
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * @file BSSNCHristoffelTilde.hpp
 * @brief Builders for the Christoffel symbols \f$\tilde{\Gamma}^i_{\ jk}\f$ and their contraction.
 * @details
 * Both helpers restrict loops to interior points so that the 4th-order derivative operators have
 * valid data.  `compute_tildeGamma_contracted` recomputes the evolved \f$\tilde{\Gamma}^i\f$ from the
 * metric divergence to enforce the algebraic definition after projection steps.  The
 * `compute_tildeGamma_symbols` utility evaluates \f$\tilde{\Gamma}^i_{\ jk}\f$ on demand without storing
 * the full 27-component tensor, keeping the solver's memory footprint minimal.
 */

namespace tensorium_RG::bssn {

/**
 * @brief Refresh the evolved \f$\tilde{\Gamma}^i\f$ from the metric inverse divergence.
 * @details Mapping to code:
 * 1. Sample `metric_inverse_divergence` (see `BSSNGamma.hpp`).
 * 2. Store \f$\tilde{\Gamma}^i = -\partial_j\tilde{\gamma}^{ij}\f$ in the SoA vector fields.
 * @see metric_inverse_divergence
 */
template <typename T> inline void compute_tildeGamma_contracted(BSSNGridSoA<T> &G) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 3, i1 = I1 - 3;
    const size_t j0 = J0 + 3, j1 = J1 - 3;
    const size_t k0 = K0 + 3, k1 = K1 - 3;

#pragma omp parallel for collapse(3)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.gamma_tilde[XX].idx(i, j, k);
                T            div[3];
                metric_inverse_divergence(G, i, j, k, div);
                G.tildeGamma[0].ptr()[id] = -div[0];
                G.tildeGamma[1].ptr()[id] = -div[1];
                G.tildeGamma[2].ptr()[id] = -div[2];
            }
}
template <typename T>
inline void compute_tildeGamma_symbols(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                       T (&Gamma)[3][3][3]) {
    using tensorium_RG::fd::Dx;
    using tensorium_RG::fd::Dy;
    using tensorium_RG::fd::Dz;

    const size_t id = G.gamma_tilde[XX].idx(i, j, k);

    auto d_g = [&](int dir, int a, int b) -> T {
        if (a > b)
            std::swap(a, b);
        const Field3D<T> &F = (a == 0 && b == 0)   ? G.gamma_tilde[XX]
                                  : (a == 0 && b == 1) ? G.gamma_tilde[XY]
                                  : (a == 0 && b == 2) ? G.gamma_tilde[XZ]
                                  : (a == 1 && b == 1) ? G.gamma_tilde[YY]
                                  : (a == 1 && b == 2) ? G.gamma_tilde[YZ]
                                                       : G.gamma_tilde[ZZ];
        return (dir == 0) ? Dx(F, i, j, k, G.dx)
               : (dir == 1) ? Dy(F, i, j, k, G.dy)
                            : Dz(F, i, j, k, G.dz);
    };

    for (int up = 0; up < 3; ++up)
        for (int lo1 = 0; lo1 < 3; ++lo1)
            for (int lo2 = 0; lo2 < 3; ++lo2) {
                T sum = T(0);
                for (int ell = 0; ell < 3; ++ell) {
                    const T ginv = sym6_get(G.gamma_tilde_inv, id, up, ell);
                    const T term = d_g(lo1, lo2, ell) + d_g(lo2, lo1, ell) - d_g(ell, lo1, lo2);
                    sum += ginv * term;
                }
                Gamma[up][lo1][lo2] = T(0.5) * sum;
            }
}

#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)

namespace detail {

template <typename T>
inline T diff_gamma_component(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k, int dir,
                              int comp_idx) {
    const Field3D<T> &F = G.gamma_tilde[comp_idx];
    if (dir == 0)
        return (T)tensorium_RG::fd::Dx(F, i, j, k, G.dx);
    if (dir == 1)
        return (T)tensorium_RG::fd::Dy(F, i, j, k, G.dy);
    return (T)tensorium_RG::fd::Dz(F, i, j, k, G.dz);
}

inline void report_validation_failure(const char *kind, size_t i, size_t j, size_t k, int a,
                                      int b, int c, double lhs, double rhs, double tol) {
    const double diff = std::abs(lhs - rhs);
    std::fprintf(stderr,
                 "[TildeGammaValidation] %s\n"
                 "\tindex=(%zu,%zu,%zu)\tcomponent=(%d,%d,%d)\n"
                 "\tlhs=% .6e\trhs=% .6e\tdiff=% .3e\ttol=% .3e\n",
                 kind, i, j, k, a, b, c, lhs, rhs, diff, tol);
    std::abort();
}

template <typename T>
inline double validation_spacing(const BSSNGridSoA<T> &G) {
    return std::max({double(G.dx), double(G.dy), double(G.dz)});
}

struct ContractedWarningState {
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t i = 0, j = 0, k = 0;
    int    comp = -1;
    double lhs = 0.0;
    double rhs = 0.0;
    size_t step = 0;
    bool   active = false;
    bool   session_open = false;
};

inline ContractedWarningState &contracted_warning_state() {
    static ContractedWarningState state;
    return state;
}

inline void begin_contracted_warning_step(size_t step = 0) {
    ContractedWarningState &state = contracted_warning_state();
    state = ContractedWarningState{};
    state.step = step;
    state.session_open = true;
}

inline void finalize_contracted_warning_step() {
    ContractedWarningState &state = contracted_warning_state();
    if (!state.session_open)
        return;
    if (state.active)
        std::fprintf(stderr,
                     "[TildeGammaValidation][WARN]\n"
                     "\tstep=%zu\tcomponent_index=%d\n"
                     "\tmax_abs=% .3e\tmax_rel=% .3e\tlocation=(%zu,%zu,%zu)\n",
                     state.step, state.comp, state.max_abs, state.max_rel, state.i, state.j,
                     state.k);
    state.session_open = false;
}

inline void update_contracted_warning_state(size_t i, size_t j, size_t k, int comp, double diff,
                                            double rel, double lhs, double rhs) {
#if defined(_OPENMP)
#    pragma omp critical(TensoriumGammaContractWarn)
#endif
    {
        ContractedWarningState &state = contracted_warning_state();
        if (!state.session_open)
            begin_contracted_warning_step();
        const bool should_update = !state.active || diff > state.max_abs || rel > state.max_rel;
        state.active = true;
        if (diff > state.max_abs)
            state.max_abs = diff;
        if (rel > state.max_rel)
            state.max_rel = rel;
        if (should_update) {
            state.i = i;
            state.j = j;
            state.k = k;
            state.comp = comp;
            state.lhs = lhs;
            state.rhs = rhs;
        }
    }
}

} // namespace detail

template <typename T>
inline void compute_tildeGamma_symbols_ref(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                           T (&Gamma)[3][3][3]) {
    T grad_metric[3][6];
    for (int dir = 0; dir < 3; ++dir)
        for (int comp = 0; comp < 6; ++comp)
            grad_metric[dir][comp] = detail::diff_gamma_component(G, i, j, k, dir, comp);

    auto grad = [&](int dir, int a, int b) -> T {
        const int idx = tensorium_RG::sym6_index(a, b);
        return grad_metric[dir][idx];
    };

    T ginv[3][3];
    const size_t id = G.gamma_tilde[XX].idx(i, j, k);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            ginv[a][b] = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);

    for (int up = 0; up < 3; ++up)
        for (int lo1 = 0; lo1 < 3; ++lo1)
            for (int lo2 = 0; lo2 < 3; ++lo2) {
                const T q0 = grad(lo1, 0, lo2) + grad(lo2, 0, lo1) - grad(0, lo1, lo2);
                const T q1 = grad(lo1, 1, lo2) + grad(lo2, 1, lo1) - grad(1, lo1, lo2);
                const T q2 = grad(lo1, 2, lo2) + grad(lo2, 2, lo1) - grad(2, lo1, lo2);
                Gamma[up][lo1][lo2] = T(0.5) * (ginv[up][0] * q0 + ginv[up][1] * q1 + ginv[up][2] * q2);
            }
}

template <typename T>
inline void validate_tildeGamma_symbols(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                        const T (&Gamma)[3][3][3]) {
    const double spacing = std::max(detail::validation_spacing(G), 1e-12);
    const double symmetry_tol = 1e-11;
    const double contracted_abs_c0 = 5e-4;
    const double contracted_rel_c1 = 5e-3;
    const double ref_tol = 2e-7 / spacing;

    const size_t id = G.gamma_tilde[XX].idx(i, j, k);

    T Gamma_ref[3][3][3];
    compute_tildeGamma_symbols_ref(G, i, j, k, Gamma_ref);

    double max_diff = 0.0;
    int    max_idx[3] = {0, 0, 0};

    for (int up = 0; up < 3; ++up)
        for (int lo1 = 0; lo1 < 3; ++lo1)
            for (int lo2 = 0; lo2 < 3; ++lo2) {
                const double sym_err = std::abs(double(Gamma[up][lo1][lo2] - Gamma[up][lo2][lo1]));
                if (lo1 <= lo2 && sym_err > symmetry_tol)
                    detail::report_validation_failure("symmetry", i, j, k, up, lo1, lo2,
                                                     Gamma[up][lo1][lo2], Gamma[up][lo2][lo1],
                                                     symmetry_tol);

                const double diff =
                    std::abs(double(Gamma[up][lo1][lo2] - Gamma_ref[up][lo1][lo2]));
                if (diff > max_diff) {
                    max_diff = diff;
                    max_idx[0] = up;
                    max_idx[1] = lo1;
                    max_idx[2] = lo2;
                }
            }

    if (max_diff > ref_tol)
        detail::report_validation_failure("ref_symbols", i, j, k, max_idx[0], max_idx[1], max_idx[2],
                                          Gamma[max_idx[0]][max_idx[1]][max_idx[2]],
                                          Gamma_ref[max_idx[0]][max_idx[1]][max_idx[2]], ref_tol);

    T contracted[3] = {T(0), T(0), T(0)};
    for (int up = 0; up < 3; ++up) {
        T sum = T(0);
        for (int lo1 = 0; lo1 < 3; ++lo1)
            for (int lo2 = 0; lo2 < 3; ++lo2) {
                const T ginv = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, lo1, lo2);
                sum += ginv * Gamma[up][lo1][lo2];
            }
        contracted[up] = sum;
    }

    T div[3];
    metric_inverse_divergence(G, i, j, k, div);
    T contracted_ref[3] = {-div[0], -div[1], -div[2]};

    for (int comp = 0; comp < 3; ++comp) {
        const double lhs = double(contracted[comp]);
        const double rhs = double(contracted_ref[comp]);
        const double diff = std::abs(lhs - rhs);
        const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
        const double tol_abs = contracted_abs_c0 * spacing * spacing;
        const double tol_rel = contracted_rel_c1 * scale;
        const double rel = diff / scale;
        if (diff > tol_abs + tol_rel)
            detail::update_contracted_warning_state(i, j, k, comp, diff, rel, lhs, rhs);
    }
}

template <typename T>
inline void compute_reference_A_up(const T (&ginv)[3][3], const T (&A_lo)[3][3],
                                   T (&A_up)[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T sum = T(0);
            for (int m = 0; m < 3; ++m)
                for (int n = 0; n < 3; ++n)
                    sum += ginv[i][m] * ginv[j][n] * A_lo[m][n];
            A_up[i][j] = sum;
        }
}

template <typename T>
inline double tensor_abs_max(const T (&M)[3][3]) {
    double maxv = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            maxv = std::max(maxv, std::abs(double(M[a][b])));
    return maxv;
}

template <typename T>
inline void validate_A_raising(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                               const T (&ginv)[3][3], const T (&A_lo)[3][3],
                               const T (&A_up_est)[3][3], T (&A_up_ref)[3][3]) {
    compute_reference_A_up(ginv, A_lo, A_up_ref);
    const double scale = std::max(1.0, tensor_abs_max(A_up_ref));
    const double tol = 5e-11 * scale;
    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
            const double lhs = double(A_up_est[a][b]);
            const double rhs = double(A_up_ref[a][b]);
            if (std::abs(lhs - rhs) > tol)
                detail::report_validation_failure("A_raise", i, j, k, a, b, -1, lhs, rhs, tol);
        }
}

template <typename T>
inline void validate_gamma_source_term(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                       const T (&Gamma_otf)[3][3][3], const T (&A_up_est)[3][3],
                                       const T (&A_up_ref)[3][3]) {
    T Gamma_ref[3][3][3];
    compute_tildeGamma_symbols_ref(G, i, j, k, Gamma_ref);

    T C_otf[3] = {T(0), T(0), T(0)};
    T C_ref[3] = {T(0), T(0), T(0)};
    for (int up = 0; up < 3; ++up)
        for (int m = 0; m < 3; ++m)
            for (int n = 0; n < 3; ++n) {
                C_otf[up] += Gamma_otf[up][m][n] * A_up_est[m][n];
                C_ref[up] += Gamma_ref[up][m][n] * A_up_ref[m][n];
            }

    const double spacing = std::max(detail::validation_spacing(G), 1e-12);
    const double A_scale = std::max(1.0, tensor_abs_max(A_up_ref));
    const double tol = (5e-7 / spacing) * A_scale;

    for (int comp = 0; comp < 3; ++comp) {
        const double lhs = double(C_otf[comp]);
        const double rhs = double(C_ref[comp]);
        if (std::abs(lhs - rhs) > tol)
            detail::report_validation_failure("GammaA", i, j, k, comp, -1, -1, lhs, rhs, tol);
    }
}

#endif // TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS

} // namespace tensorium_RG::bssn
