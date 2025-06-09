#pragma once

#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNAutoDiff.hpp"
#include "BSSNChiContext.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNContractedChristoffel.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {

template <typename T> class RicciConformalTensor {
  public:
    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_Hessian(const ChiContext<T>           &chi_context,
                              const tensorium::Tensor<T, 2> &gamma_tilde,
                              const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                              const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;
        Tensor<T, 2> result({3, 3});

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T corrected_hessian = chi_context.hessian_chi(i, j);
                for (int k = 0; k < 3; ++k)
                    corrected_hessian -= christoffel_tilde(k, i, j) * chi_context.grad_chi(k);
                result(i, j) = corrected_hessian / chi_context.chi;
            }
        }

        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) {
                T avg = 0.5 * (result(i, j) + result(j, i));
                result(i, j) = result(j, i) = avg;
            }

        return result;
    }

    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_Laplacian(const ChiContext<T>           &chi_context,
                                const tensorium::Tensor<T, 2> &gamma_tilde,
                                const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                                const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;

        Tensor<T, 2> result({3, 3});

        const auto &grad_chi = chi_context.grad_chi;
        const auto &hessian_chi = chi_context.hessian_chi;
        const T     chi = chi_context.chi;

        T laplacian_chi = 0.0;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                T term = hessian_chi(k, l);
                for (int m = 0; m < 3; ++m)
                    term -= christoffel_tilde(m, k, l) * grad_chi(m);
                laplacian_chi += gamma_tilde_inv(k, l) * term;
            }
        }

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = 0.5 * gamma_tilde(i, j) * laplacian_chi / chi;

        return result;
    }

    static tensorium::Tensor<T, 2> compute_Ricci_chi_GradGrad(const ChiContext<T> &chi_context) {
        using namespace tensorium;

        Tensor<T, 2> result({3, 3});

        const auto &grad_chi = chi_context.grad_chi;
        const T     chi = chi_context.chi;

        const T inv_chi2 = 1.0 / (chi * chi);
        const T coeff = -1.5 * inv_chi2;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = coeff * grad_chi(i) * grad_chi(j);

        return result;
    }

    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_GradNorm(const ChiContext<T>           &chi_context,
                               const tensorium::Tensor<T, 2> &gamma_tilde,
                               const tensorium::Tensor<T, 2> &gamma_tilde_inv) {
        using namespace tensorium;

        Tensor<T, 2> result({3, 3});

        const auto &grad_chi = chi_context.grad_chi;
        const T     chi = chi_context.chi;
        const T     inv_chi2 = 1.0 / (chi * chi);

        T grad_norm = 0.0;
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l)
                grad_norm += gamma_tilde_inv(k, l) * grad_chi(k) * grad_chi(l);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = 0.5 * inv_chi2 * gamma_tilde(i, j) * grad_norm;

        return result;
    }

    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_total(const ChiContext<T>           &chi_context,
                            const tensorium::Tensor<T, 2> &gamma_tilde,
                            const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                            const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;

        Tensor<T, 2> term1 =
            compute_Ricci_chi_Hessian(chi_context, gamma_tilde, gamma_tilde_inv, christoffel_tilde);
        Tensor<T, 2> term2 = compute_Ricci_chi_Laplacian(chi_context, gamma_tilde, gamma_tilde_inv,
                                                         christoffel_tilde);
        Tensor<T, 2> term3 = compute_Ricci_chi_GradGrad(chi_context);
        Tensor<T, 2> term4 = compute_Ricci_chi_GradNorm(chi_context, gamma_tilde, gamma_tilde_inv);

        Tensor<T, 2> result({3, 3});
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = term1(i, j) + term2(i, j) + term3(i, j) + term4(i, j);

        return result;
    }
};
} // namespace tensorium_RG
