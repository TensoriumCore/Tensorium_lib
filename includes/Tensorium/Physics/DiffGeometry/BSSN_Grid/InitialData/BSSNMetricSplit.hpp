#pragma once

#include "../../Metric.hpp"
#include "../Fields/BSSNTypes.hpp"
#include <cmath>

/**
 * @file BSSNMetricSplit.hpp
 * @brief Convert analytic spacetime metrics into ADM/BSSN variables sampled at grid points.
 */

namespace tensorium_RG::bssn {

/**
 * @brief Evaluate the ADM 3+1 split of `metric` at spatial point `X`.
 */
template <typename T>
inline ADMVariables<T> split_3p1(const tensorium::Vector<T> &X, const Metric<T> &metric) {
    ADMVariables<T> adm;
    metric.BSSN(X, adm.alpha, adm.beta, adm.gamma_ij);
    return adm;
}

/**
 * @brief Convert ADM spatial metric \f$\gamma_{ij}\f$ into conformal variables \f$\chi,\tilde{\gamma}_{ij}\f$.
 */
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
