#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace tensorium_RG {


	template<typename T>
		class TildeGamma {
			public:
				/**
				 * @brief Compute the vector \tilde{\Gamma}^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}
				 * 
				 * @param gamma_tilde_inv    3×3 inverse conformal metric \tilde{\gamma}^{ij}
				 * @param christoffel        3×3×3 conformal Christoffel symbols \tilde{\Gamma}^i_{jk}
				 * @param tildeGamma         Output vector \tilde{\Gamma}^i
				 */
				static void compute(
						const tensorium::Tensor<T, 2>& gamma_tilde_inv,
						const tensorium::Tensor<T, 3>& christoffel,
						tensorium::Vector<T>& tildeGamma
						) {
					if (gamma_tilde_inv.shape() != std::array<size_t, 2>{3, 3})
						throw std::invalid_argument("gamma_tilde_inv must be 3x3");
					if (christoffel.shape() != std::array<size_t, 3>{3, 3, 3})
						throw std::invalid_argument("christoffel must be 3x3x3");

					tildeGamma.resize(3);
					for (size_t i = 0; i < 3; ++i) {
						T sum = 0.0;
						for (size_t j = 0; j < 3; ++j) {
							for (size_t k = 0; k < 3; ++k) {
								sum += gamma_tilde_inv(j, k) * christoffel(i, j, k);
							}
						}
						tildeGamma[i] = sum;
					}
				}
		};
}
