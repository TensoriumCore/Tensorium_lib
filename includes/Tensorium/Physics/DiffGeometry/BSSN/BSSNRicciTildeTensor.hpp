/**
 * @file BSSNRicciTildeTensor.hpp
 * @brief Computes the conformal part \f$ \tilde{R}_{ij} \f$ of the Ricci tensor in the BSSN
 * formalism.
 */

#pragma once

#include "../../../Core/Matrix.hpp"
#include "../../../Core/Tensor.hpp"
#include "../../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNAutoDiff.hpp"
#include "BSSNChiContext.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNContractedChristoffel.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {
/**
 * @class RicciTildeTensor
 * @brief Provides methods to compute the conformal Ricci tensor \f$ \tilde{R}_{ij} \f$ in the BSSN
 * formulation.
 *
 * The conformal Ricci tensor is given by:
 * \f[
 * \tilde{R}_{ij} = R^{(1)}_{ij} + R^{(2)}_{ij} + R^{(3)}_{ij} + R^{(4)}_{ij}
 * \f]
 *
 * where:
 * - \f$ R^{(1)}_{ij} = -\frac{1}{2} \tilde{\gamma}^{kl} \partial_k \partial_l \tilde{\gamma}_{ij}
 * \f$
 * - \f$ R^{(2)}_{ij} = \frac{1}{2} \left( \nabla_i \tilde{\Gamma}_j + \nabla_j \tilde{\Gamma}_i
 * \right) \f$
 * - \f$ R^{(3)}_{ij} = \frac{1}{2} \tilde{\Gamma}^k \left( \tilde{\gamma}_{ki}
 * \tilde{\Gamma}^l_{jl} + \tilde{\gamma}_{kj} \tilde{\Gamma}^l_{il} \right) \f$
 * - \f$ R^{(4)}_{ij} = \tilde{\gamma}^{\ell m} \left( 2 \tilde{\Gamma}^k_{\ell(i}
 * \tilde{\Gamma}_{j)km} + \tilde{\Gamma}^k_{im} \tilde{\Gamma}_{k\ell j} \right) \f$
 *
 * Each term is computed separately and combined to give \f$ \tilde{R}_{ij} \f$.
 */
template <typename T> class RicciTildeTensor {
  public:
    /**
     * @brief Compute \f$ R^{(1)}_{ij} = -\frac{1}{2} \tilde{\gamma}^{kl} \partial_k \partial_l
     * \tilde{\gamma}_{ij} \f$ using scalar autodiff.
     *
     * @param X               Position vector
     * @param dx,dy,dz        Grid spacings
     * @param metric          Metric object to evaluate \f$ \tilde{\gamma}_{ij} \f$
     * @param gamma_tilde_inv Inverse of the conformal metric \f$ \tilde{\gamma}^{kl} \f$
     * @return Laplacian term \f$ R^{(1)}_{ij} \f$
     */

    static tensorium::Tensor<T, 2>
    compute_laplacian_term(const tensorium::Vector<T> &X, T dx, T dy, T dz, const Metric<T> &metric,
                           const tensorium::Tensor<T, 2> &gamma_tilde_inv) {
        using namespace tensorium;

        Tensor<T, 2> R1({3, 3});

        const int off = (X.size() == 4) ? 1 : 0;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {

                auto gij = [&](const Vector<T> &Xs) -> T {
                    T            alpha;
                    Vector<T>    beta(3);
                    Tensor<T, 2> g_phys({3, 3});
                    metric.BSSN(Xs, alpha, beta, g_phys);

                    T            chi = compute_conformal_factor(metric, g_phys);
                    Tensor<T, 2> gtilde = compute_conformal_metric(metric, g_phys, chi);
                    return gtilde(i, j);
                };

                Tensor<T, 2> hess({3, 3});

                if (off == 0) {
                    compute_second_derivatives_scalar(X, dx, dy, dz, gij, hess);
                } else {
                    Vector<T> X3(3);
                    X3(0) = X(1);
                    X3(1) = X(2);
                    X3(2) = X(3);

                    auto gij_from_X3 = [&](const Vector<T> &X3s) -> T {
                        Vector<T> X4 = X;
                        X4(1) = X3s(0);
                        X4(2) = X3s(1);
                        X4(3) = X3s(2);
                        return gij(X4);
                    };

                    compute_second_derivatives_scalar(X3, dx, dy, dz, gij_from_X3, hess);
                }

                T lap = T(0);
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        lap += gamma_tilde_inv(k, l) * hess(k, l);

                R1(i, j) = T(-0.5) * lap;
            }
        }

        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) {
                T avg = T(0.5) * (R1(i, j) + R1(j, i));
                R1(i, j) = R1(j, i) = avg;
            }

        return R1;
    }

    /**
     * @brief Compute \f$ R^{(2)}_{ij} = \frac{1}{2} \left( \nabla_i \tilde{\Gamma}_j + \nabla_j
     * \tilde{\Gamma}_i \right) \f$
     *
     * @param X                 Spatial point
     * @param dx, dy, dz        Grid spacings
     * @param tilde_Gamma       Contracted conformal Christoffel vector \f$ \tilde{\Gamma}^i \f$
     * @param tilde_gamma       Conformal metric \f$ \tilde{\gamma}_{ij} \f$
     * @return Symmetrized derivative term \f$ R^{(2)}_{ij} \f$
     */
    static tensorium::Tensor<T, 2>
    compute_dGamma_term(const tensorium::Vector<T> &X, T dx, T dy, T dz,
                        const tensorium::Vector<T>    &tilde_Gamma,
                        const tensorium::Tensor<T, 2> &tilde_gamma_contract) {
        using namespace tensorium;

        Tensor<T, 2> dGamma({3, 3});
        for (int k = 0; k < 3; ++k) {
            auto      scalar_func = [&](const Vector<T> &Xs) -> T { return tilde_Gamma(k); };
            Vector<T> grad_k = partial_scalar(X, dx, dy, dz, scalar_func);
            for (int j = 0; j < 3; ++j) {
                dGamma(j, k) = grad_k(j);
            }
        }

        Tensor<T, 2> R2({3, 3});
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T sum = 0;
                for (int k = 0; k < 3; ++k) {
                    sum += tilde_gamma_contract(k, i) * dGamma(j, k) +
                           tilde_gamma_contract(k, j) * dGamma(i, k);
                }
                R2(i, j) = static_cast<T>(0.5) * sum;
            }
        }

        return R2;
    }

    /**
     * @brief Compute the nonlinear contracted term:
     * \f[
     * R^{(3)}_{ij} = \frac{1}{2} \tilde{\Gamma}^k \left( \tilde{\gamma}_{ki} \tilde{\Gamma}^l_{jl}
     * + \tilde{\gamma}_{kj} \tilde{\Gamma}^l_{il} \right)
     * \f]
     *
     * @param tilde_Gamma        Contracted Christoffel symbols \f$ \tilde{\Gamma}^k \f$
     * @param christoffel_tilde  Christoffel symbols \f$ \tilde{\Gamma}^i_{jk} \f$
     * @param gamma_tilde        Conformal metric \f$ \tilde{\gamma}_{ij} \f$
     * @return Tensor \f$ R^{(3)}_{ij} \f$
     */
    static tensorium::Tensor<T, 2>
    compute_GammaGamma_term(const tensorium::Vector<T>    &tilde_Gamma,
                            const tensorium::Tensor<T, 3> &christoffel_tilde,
                            const tensorium::Tensor<T, 2> &gamma_tilde) {
        using namespace tensorium;
        Tensor<T, 2> R3({3, 3});

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                T sum_ij = 0;
                for (size_t k = 0; k < 3; ++k) {
                    T G_ijk = 0, G_jik = 0;
                    for (size_t m = 0; m < 3; ++m) {
                        G_ijk += gamma_tilde(i, m) * christoffel_tilde(m, j, k);
                        G_jik += gamma_tilde(j, m) * christoffel_tilde(m, i, k);
                    }
                    sum_ij += tilde_Gamma(k) * (G_ijk + G_jik);
                }
                R3(i, j) = T(0.5) * sum_ij;
            }
        }

        return R3;
    }

    /**
     * @brief Compute the quadratic Christoffel term:
     * \f[
     * R^{(4)}_{ij} = \tilde{\gamma}^{\ell m} \left( 2 \tilde{\Gamma}^k_{\ell(i}
     * \tilde{\Gamma}_{j)km} + \tilde{\Gamma}^k_{im} \tilde{\Gamma}_{k\ell j} \right)
     * \f]
     *
     * @param gamma_tilde_inv    Inverse conformal metric \f$ \tilde{\gamma}^{\ell m} \f$
     * @param christoffel_tilde  Christoffel symbols \f$ \tilde{\Gamma}^i_{jk} \f$
     * @return Tensor \f$ R^{(4)}_{ij} \f$
     */
    static tensorium::Tensor<T, 2>
    compute_GammaProduct_term(const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                              const tensorium::Tensor<T, 3> &christoffel_tilde,
                              const tensorium::Tensor<T, 2> &gamma_tilde) {
        using namespace tensorium;
        Tensor<T, 2> R4({3, 3});

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                T sum = 0;
                for (size_t ell = 0; ell < 3; ++ell) {
                    for (size_t m = 0; m < 3; ++m) {
                        T g_inv = gamma_tilde_inv(ell, m);
                        T term_k_sum = 0;

                        for (size_t k = 0; k < 3; ++k) {
                            T Gamma_jkm = 0;
                            T Gamma_ikm = 0;
                            for (size_t p = 0; p < 3; ++p) {
                                Gamma_jkm += gamma_tilde(j, p) * christoffel_tilde(p, k, m);
                                Gamma_ikm += gamma_tilde(i, p) * christoffel_tilde(p, k, m);
                            }

                            T Gamma_klj = 0;
                            for (size_t p = 0; p < 3; ++p) {
                                Gamma_klj += gamma_tilde(k, p) * christoffel_tilde(p, ell, j);
                            }

                            T t1 = christoffel_tilde(k, ell, i) * Gamma_jkm;
                            T t2 = christoffel_tilde(k, ell, j) * Gamma_ikm;

                            T t3 = christoffel_tilde(k, i, m) * Gamma_klj;

                            term_k_sum += (t1 + t2 + t3);
                        }
                        sum += g_inv * term_k_sum;
                    }
                }
                R4(i, j) = sum;
            }
        }
        return R4;
    } /**
       * @brief Combine all four contributions to compute the conformal Ricci tensor:
       * \f[
       * \tilde{R}_{ij} = R^{(1)}_{ij} + R^{(2)}_{ij} + R^{(3)}_{ij} + R^{(4)}_{ij}
       * \f]
       *
       * @param chi_context        ChiContext (holds metric, position, grid spacing, etc.)
       * @param gamma_tilde_inv    Inverse conformal metric
       * @param tilde_Gamma        Contracted conformal Christoffel vector
       * @param christoffel_tilde  Conformal Christoffel symbols
       * @param gamma_tilde        Conformal metric
       * @return Tensor \f$ \tilde{R}_{ij} \f$
       */
    static tensorium::Tensor<T, 2> compute_Ricci_Tilde_tensor(
        const ChiContext<T> &chi_context, const tensorium::Tensor<T, 2> &gamma_tilde_inv,
        const tensorium::Vector<T> &tilde_Gamma, const tensorium::Tensor<T, 3> &christoffel_tilde,
        const tensorium::Tensor<T, 2> &gamma_tilde) {
        using namespace tensorium;
        const auto &X = chi_context.X;
        const T     dx = chi_context.dx;
        const T     dy = chi_context.dy;
        const T     dz = chi_context.dz;
        const auto &metric = chi_context.metric;
        auto        R1 = compute_laplacian_term(X, dx, dy, dz, metric, gamma_tilde_inv);
        auto        R2 = compute_dGamma_term(X, dx, dy, dz, tilde_Gamma, gamma_tilde);
        auto        R3 = compute_GammaGamma_term(tilde_Gamma, christoffel_tilde, gamma_tilde);
        auto        R4 = compute_GammaProduct_term(gamma_tilde_inv, christoffel_tilde, gamma_tilde);
        tensorium::Tensor<T, 2> Ricci({3, 3});
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                Ricci(i, j) = R1(i, j) + R2(i, j) + R3(i, j) + R4(i, j);
            }
        }

        return Ricci;
    }
};

} // namespace tensorium_RG
