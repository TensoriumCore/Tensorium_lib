#pragma once

#include "../Metric.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../../Core/Derivate.hpp"
#include "BSSNSetup.hpp"

namespace tensorium_RG {

	template<typename K>
		class ExtrinsicCurvature {
			public:
				using Vec = tensorium::Vector<K>;
				using Mat = tensorium::Tensor<K, 2>;
				

				template<typename T>
					tensorium::Tensor<T, 2> compute_Kij(
							const tensorium::Tensor<T, 2>& dgt,
							const tensorium::Tensor<T, 2>& gamma,
							const tensorium::Vector<T>& beta,
							const tensorium::Tensor<T, 2>& partial_beta,
							const tensorium::Tensor<T, 3>& christoffel,
							const T alpha
							) {
						tensorium::Tensor<T, 2> Kij({3, 3});

						for (size_t i = 0; i < 3; ++i) {
							for (size_t j = 0; j < 3; ++j) {
								T sym_grad_beta = partial_beta(i, j) + partial_beta(j, i);
								T gamma_beta = T(0.0);
								for (size_t k = 0; k < 3; ++k)
									gamma_beta += 2.0 * christoffel(i, j, k) * beta(k);

								Kij(i, j) = -0.5 / alpha * (dgt(i, j) - (sym_grad_beta - gamma_beta));
							}
						}
						for (size_t i = 0; i < 3; ++i)
							for (size_t j = i+1; j < 3; ++j) {
								T sym_val = 0.5 * (Kij(i, j) + Kij(j, i));
								Kij(i, j) = Kij(j, i) = sym_val;
							}


						return Kij;
					}
		};

}
