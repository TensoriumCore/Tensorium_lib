#pragma once

#include "../Core/Derivate.hpp"
#include "../Core/Matrix.hpp"
#include "../Core/Tensor.hpp"
#include "../Core/Vector.hpp"
#include "../DiffGeometry/ChristoffelSymbol.hpp"
#include "../DiffGeometry/Metric.hpp"
namespace tensorium_RG {
/**
 * @brief Computes the Ricci tensor and Ricci scalar from a 4D Riemann tensor.
 *
 * This class provides static methods to contract the Riemann tensor into the Ricci tensor,
 * compute the scalar curvature, and print these results.
 *
 * The Ricci tensor is obtained by contraction:
 * \f[
 * R_{\mu\nu} = R^\rho_{\ \mu\rho\nu} = g^{\rho\sigma} R_{\rho\sigma\mu\nu}
 * \f]
 *
 * The Ricci scalar is the trace:
 * \f[
 * R = g^{\mu\nu} R_{\mu\nu}
 * \f]
 */
template <typename T> class RicciTensor {
  public:
    using Tensor4D = tensorium::Tensor<T, 4>;
    using Tensor2D = tensorium::Tensor<T, 2>;
    using VectorT = tensorium::Vector<T>;
    /**
     * @brief Contracts a 4D Riemann tensor into the 2D Ricci tensor.
     *
     * Uses the identity \f$ R_{\mu\nu} = g^{\rho\sigma} R_{\rho\sigma\mu\nu} \f$
     *
     * @param Riemann The Riemann tensor \f$ R_{\rho\sigma\mu\nu} \f$
     * @param g_inv The inverse metric \f$ g^{\rho\sigma} \f$
     * @return The Ricci tensor \f$ R_{\mu\nu} \f$
     */
    static tensorium::Tensor<T, 2> contract_to_ricci(const Tensor4D                &Riemann,
                                                     const tensorium::Tensor<T, 2> &g_inv) {
        constexpr size_t        dim = 4;
        tensorium::Tensor<T, 2> Ricci({dim, dim});
        Ricci.fill(T(0));

        for (size_t mu = 0; mu < dim; ++mu) {
            for (size_t nu = 0; nu < dim; ++nu) {
                for (size_t rho = 0; rho < dim; ++rho) {
                    for (size_t sigma = 0; sigma < dim; ++sigma) {
                        Ricci(mu, nu) += g_inv(rho, sigma) * Riemann(rho, sigma, mu, nu);
                    }
                }
            }
        }

        return Ricci;
    }
    /**
     * @brief Computes the Ricci scalar curvature.
     *
     * Uses the trace formula: \f$ R = g^{\mu\nu} R_{\mu\nu} \f$
     *
     * @param Ricci The Ricci tensor \f$ R_{\mu\nu} \f$
     * @param g_inv The inverse metric \f$ g^{\mu\nu} \f$
     * @return The scalar curvature \f$ R \f$
     */
    static double compute_ricci_scalar(const Tensor2D                &Ricci,
                                       const tensorium::Tensor<T, 2> &g_inv) {
        constexpr size_t dim = 4;
        double           R = 0.0;
        for (size_t mu = 0; mu < dim; ++mu) {
            for (size_t nu = 0; nu < dim; ++nu) {
                R += g_inv(mu, nu) * Ricci(mu, nu);
            }
        }
        return R;
    }
    /**
     * @brief Prints the Ricci tensor components with optional zero threshold.
     *
     * @param R The Ricci tensor
     * @param threshold Minimum value to display (default = 1e-12)
     */
    static void print_componentwise(const Tensor2D &R, T threshold = 1e-12) {
        constexpr size_t dim = 4;
        std::cout << "\nRicci tensor components:\n";
        for (size_t mu = 0; mu < dim; ++mu) {
            for (size_t nu = 0; nu < dim; ++nu) {
                T val = R(mu, nu);
                if (std::abs(val) < threshold)
                    val = 0.0;
                std::cout << std::setw(10) << std::fixed << std::setprecision(6) << val << " ";
            }
            std::cout << "\n";
        }
    }

    /**
     * @brief Prints only the non-zero components of the Ricci tensor.
     *
     * @param R The Ricci tensor
     * @param name Optional tensor name label
     */
    static void print(const Tensor2D &R, const std::string &name = "Ricci") {
        constexpr size_t dim = 4;
        std::cout << name << " tensor components:\n";
        for (size_t mu = 0; mu < dim; ++mu)
            for (size_t nu = 0; nu < dim; ++nu) {
                const T val = R(mu, nu);
                if (std::abs(val) > T(1e-12)) {
                    std::cout << name << "[" << mu << "][" << nu << "] = " << val << "\n";
                }
            }
    }
    /**
     * @brief Prints non-zero components using both Ricci and inverse metric.
     *
     * @param R The Ricci tensor
     * @param g_inv The inverse metric
     * @param name Optional tensor name label
     */
    static void print(const Tensor2D &R, const tensorium::Tensor<T, 2> &g_inv,
                      const std::string &name = "Ricci") {
        constexpr size_t dim = 4;
        std::cout << name << " tensor components:\n";
        for (size_t mu = 0; mu < dim; ++mu)
            for (size_t nu = 0; nu < dim; ++nu) {
                const T val = R(mu, nu);
                if (std::abs(val) > T(1e-12)) {
                    std::cout << name << "[" << mu << "][" << nu << "] = " << val << "\n";
                }
            }
    }

    /**
     * @brief Computes and prints the Ricci scalar from the tensor and metric.
     *
     * @param Ricci The Ricci tensor
     * @param g_inv The inverse metric
     */
    static void print_ricci_scalar(const Tensor2D &Ricci, const tensorium::Tensor<T, 2> &g_inv) {
        constexpr size_t dim = 4;
        double           R = compute_ricci_scalar(Ricci, g_inv);
        std::cout << "Ricci scalar R = " << R << "\n";
    }
};
} // namespace tensorium_RG
