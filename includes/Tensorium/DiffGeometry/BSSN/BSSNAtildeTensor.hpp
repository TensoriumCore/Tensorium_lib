#pragma once

#include "../Metric.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../../Core/Derivate.hpp"
#include "BSSNSetup.hpp"

namespace tensorium_RG {

	template<typename K>
	class BSSNAtildeTensor {
	public:
		using Vec = tensorium::Vector<K>;
		using Mat = tensorium::Tensor<K, 2>;

		/**
		 * @brief Compute the trace-free conformal extrinsic curvature tensor Ã_ij
		 *        Ã_ij = χ (K_ij - (1/3) γ_ij K)
		 *
		 * @param Kij        : extrinsic curvature tensor K_ij
		 * @param gamma_inv  : inverse spatial metric γ^{ij}
		 * @param gamma      : spatial metric γ_ij
		 * @param chi        : conformal factor
		 * @return tensorium::Tensor<K, 2> Ã_ij
		 */
		tensorium::Tensor<K, 2> compute_Atilde_tensor(const tensorium::Tensor<K, 2>& Kij,
										   const tensorium::Tensor<K, 2>& gamma_inv,
										   const tensorium::Tensor<K, 2>& gamma,
										   K chi) {
			K trace_K = K(0);
			for (size_t i = 0; i < 3; ++i)
				for (size_t j = 0; j < 3; ++j)
					trace_K += gamma_inv(i, j) * Kij(i, j);

			tensorium::Tensor<K, 2> Atilde({3, 3});
			for (size_t i = 0; i < 3; ++i)
				for (size_t j = 0; j < 3; ++j)
					Atilde(i, j) = chi * (Kij(i, j) - (1. / 3.) * gamma(i, j) * trace_K);

			// Ensure symmetry: Ã_ij = Ã_ji
			for (size_t i = 0; i < 3; ++i)
				for (size_t j = i + 1; j < 3; ++j) {
					K sym = (Atilde(i, j) + Atilde(j, i)) * 0.5;
					Atilde(i, j) = Atilde(j, i) = sym;
				}

			return Atilde;
		}
	};

}
