
#pragma once
#include "../../../../Core/Tensor.hpp"
#include "../../../../Core/Vector.hpp"

/**
 * @file BSSNTypes.hpp
 * @brief Lightweight POD structs for ADM and BSSN variables used when importing external metrics.
 * @details
 * Initial-data utilities (e.g., metric splitting, Bowen–York solvers) use these containers to pass
 * \f$(\alpha,\beta^i,\gamma_{ij})\f$ tuples through analytic construction code before populating the
 * structure-of-arrays grid.  The tensors employ row-major storage consistent with the rest of the
 * Tensorium core.
 */

namespace tensorium_RG::bssn {

/**
 * @brief ADM \f$(3+1)\f$ variables evaluated at a spatial point.
 * @tparam T Arithmetic type; defaults to double when sampling analytic spacetimes.
 */
template <typename T> struct ADMVariables {
    T                       alpha;
    tensorium::Vector<T>    beta;     // size 3
    tensorium::Tensor<T, 2> gamma_ij; // 3x3

    ADMVariables() : beta(3), gamma_ij({3, 3}) {}
};

/**
 * @brief Conformal variables \f$\{\chi,\tilde{\gamma}_{ij},\tilde{\gamma}^{ij}\}\f$ derived from ADM data.
 */
template <typename T> struct BSSNVariables {
    T                       chi;
    tensorium::Tensor<T, 2> gamma_tilde; // 3x3
    tensorium::Tensor<T, 2> gamma_tilde_inv;

    BSSNVariables() : gamma_tilde({3, 3}), gamma_tilde_inv({3, 3}) {}
};

template <typename T> using Z4cADMVariables = ADMVariables<T>;
template <typename T> using Z4cVariables = BSSNVariables<T>;

} // namespace tensorium_RG::bssn
