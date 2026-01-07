#pragma once

#include "../Metric.hpp"
#include "BSSNTypes.hpp"
#include <cmath>

namespace tensorium_RG::bssn {

template <typename T>
inline ADMVariables<T> split_3p1(const tensorium::Vector<T> &X, const Metric<T> &metric) {
    ADMVariables<T> adm;
    metric.BSSN(X, adm.alpha, adm.beta, adm.gamma_ij);
    return adm;
}

template <typename T> inline BSSNVariables<T> adm_to_bssn(const tensorium::Tensor<T, 2> &gamma_ij) {
    BSSNVariables<T> bssn;

    const T det = det_tensor(gamma_ij);
    bssn.chi = std::pow(det, T(-1.0 / 3.0));

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            bssn.gamma_tilde(i, j) = bssn.chi * gamma_ij(i, j);

    bssn.gamma_tilde_inv = inv_mat_tensor(bssn.gamma_tilde);
    return bssn;
}

} // namespace tensorium_RG::bssn
