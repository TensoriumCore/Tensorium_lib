#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "BSSNRicciConformalTensor.hpp"
#include "BSSNRicciTildeTensor.hpp"

namespace tensorium_RG {

template <typename T> class RicciPhysicalTensor {
  public:
    static tensorium::Tensor<T, 2> compute_Ricci_total(
        const ChiContext<T> &chi_context, const tensorium::Tensor<T, 2> &gamma_tilde,
        const tensorium::Tensor<T, 2> &gamma_tilde_inv, const tensorium::Vector<T> &tilde_Gamma,
        const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;

        Tensor<T, 2> Ricci_tilde = RicciTildeTensor<T>::compute_Ricci_Tilde_tensor(
            chi_context, gamma_tilde_inv, tilde_Gamma, christoffel_tilde, gamma_tilde);

        Tensor<T, 2> Ricci_chi = RicciConformalTensor<T>::compute_Ricci_chi_total(
            chi_context, gamma_tilde, gamma_tilde_inv, christoffel_tilde);

        Tensor<T, 2> Ricci_phys({3, 3});
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Ricci_phys(i, j) = Ricci_tilde(i, j) + Ricci_chi(i, j);

        T norm = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                norm += Ricci_phys(i, j) * Ricci_phys(i, j);

        norm = std::sqrt(norm);
        std::cout << "||R_phys||_F = " << norm << std::endl;

        return Ricci_phys;
    }
};

} // namespace tensorium_RG
