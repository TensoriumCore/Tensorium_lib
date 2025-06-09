#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNSetup.hpp"

namespace tensorium_RG {

template <typename T>
inline void print_tensor2(const std::string &name, const tensorium::Tensor<T, 2> &tensor) {
    std::cout << "--- " << name << " ---\n";
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j)
            std::cout << std::setw(12) << tensor(i, j) << " ";
        std::cout << "\n";
    }
}

template <typename T>
inline void print_tensor3(const std::string &name, const tensorium::Tensor<T, 3> &tensor) {
    std::cout << "--- " << name << " ---\n";
    for (size_t d = 0; d < 3; ++d) {
        std::cout << "∂_" << "xyz"[d] << ":\n";
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j)
                std::cout << std::setw(12) << tensor(d, i, j) << " ";
            std::cout << "\n";
        }
    }
}

template <typename T>
inline void print_vector(const std::string &name, const tensorium::Vector<T> &vec) {
    std::cout << "--- " << name << " ---\n";
    for (size_t i = 0; i < vec.size(); ++i)
        std::cout << std::setw(12) << vec(i) << " ";
    std::cout << "\n";
}

} // namespace tensorium_RG
