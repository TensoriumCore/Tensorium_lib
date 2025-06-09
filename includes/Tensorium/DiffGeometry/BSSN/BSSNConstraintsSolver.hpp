#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include <cassert>
#include <cstddef>

namespace tensorium_RG {
template <typename T> class ConstraintSolver {
  public:
    /**
     * @brief Solve the Lichnerowicz equation to enforce the Hamiltonian constraint in BSSN.
     *
     * @param Atilde        The trace-free conformal extrinsic curvature tensor Ã_{ij} at each grid
     * point. Shape: [NX][NY][NZ][3][3]
     * @param g_tilde_inv   The inverse conformal metric γ̃^{ij} at each grid point.
     *                      Shape: [NX][NY][NZ][3][3]
     * @param dx, dy, dz    Grid spacings
     * @param max_iter      Maximum number of iterations
     * @param tol           Convergence threshold
     * @return psi          The conformal factor ψ at each grid point. Shape: [NX][NY][NZ]
     */
    static tensorium::Tensor<T, 3> solveLichnerowicz(const tensorium::Tensor<T, 5> &Atilde,
                                                     const tensorium::Tensor<T, 5> &g_tilde_inv,
                                                     T dx, T dy, T dz, int max_iter = 1000,
                                                     T tol = T(1e-8)) {
        std::array<size_t, 5> shp = Atilde.shape();
        const size_t          NX = shp[0];
        const size_t          NY = shp[1];
        const size_t          NZ = shp[2];

        tensorium::Tensor<T, 3> psi({NX, NY, NZ});
        for (size_t i = 0; i < NX; ++i) {
            for (size_t j = 0; j < NY; ++j) {
                for (size_t k = 0; k < NZ; ++k) {
                    psi({i, j, k}) = T(1);
                }
            }
        }

        const T inv_dx2 = T(1) / (dx * dx);
        const T inv_dy2 = T(1) / (dy * dy);
        const T inv_dz2 = T(1) / (dz * dz);
        const T factor = T(1) / (T(2) * (inv_dx2 + inv_dy2 + inv_dz2));

        for (int it = 0; it < max_iter; ++it) {
            T max_diff = T(0);

            for (size_t i = 1; i < NX - 1; ++i) {
                for (size_t j = 1; j < NY - 1; ++j) {
                    for (size_t k = 1; k < NZ - 1; ++k) {
                        T lap =
                            inv_dx2 *
                                (psi({i + 1, j, k}) + psi({i - 1, j, k}) - T(2) * psi({i, j, k})) +
                            inv_dy2 *
                                (psi({i, j + 1, k}) + psi({i, j - 1, k}) - T(2) * psi({i, j, k})) +
                            inv_dz2 *
                                (psi({i, j, k + 1}) + psi({i, j, k - 1}) - T(2) * psi({i, j, k}));

                        T A2 = T(0);
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                for (int c = 0; c < 3; ++c) {
                                    for (int d = 0; d < 3; ++d) {
                                        // accès à g_tilde_inv et Atilde par array d’indices
                                        T g1 = g_tilde_inv({i, j, k, (size_t)a, (size_t)c});
                                        T g2 = g_tilde_inv({i, j, k, (size_t)b, (size_t)d});
                                        T A3 = Atilde({i, j, k, (size_t)c, (size_t)d});
                                        T A4 = Atilde({i, j, k, (size_t)a, (size_t)b});
                                        A2 += g1 * g2 * A3 * A4;
                                    }
                                }
                            }
                        }

                        T psi_ijk = std::fmax(psi({i, j, k}), T(1e-8));
                        T rhs = (T(1) / T(8)) * A2 / std::pow(psi_ijk, T(7));

                        T new_psi = psi({i, j, k}) + T(0.5) * (lap - rhs) * factor;
                        new_psi = std::fmax(new_psi, T(0.1));

                        max_diff = std::fmax(max_diff, std::abs(new_psi - psi({i, j, k})));
                        psi({i, j, k}) = new_psi;
                    }
                }
            }

            if (max_diff < tol) {
                break;
            }
        }

        return psi;
    }
};

} // namespace tensorium_RG
