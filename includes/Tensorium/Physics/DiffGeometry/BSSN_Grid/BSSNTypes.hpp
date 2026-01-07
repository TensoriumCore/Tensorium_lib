
#pragma once
#include "../../../Core/Tensor.hpp"
#include "../../../Core/Vector.hpp"

namespace tensorium_RG::bssn {

template <typename T> struct ADMVariables {
    T                       alpha;
    tensorium::Vector<T>    beta;     // size 3
    tensorium::Tensor<T, 2> gamma_ij; // 3x3

    ADMVariables() : beta(3), gamma_ij({3, 3}) {}
};

template <typename T> struct BSSNVariables {
    T                       chi;
    tensorium::Tensor<T, 2> gamma_tilde; // 3x3
    tensorium::Tensor<T, 2> gamma_tilde_inv;

    BSSNVariables() : gamma_tilde({3, 3}), gamma_tilde_inv({3, 3}) {}
};

} // namespace tensorium_RG::bssn
