#pragma once

#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Vector.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {

	template<typename T>
		class RicciTensor3D {
			public:

				static tensorium::Tensor<T, 5> compute_ricci_conformal(
						const tensorium::Tensor<T, 5>& gamma_tilde,
						const tensorium::Tensor<T, 5>& gamma_tilde_inv,
						const tensorium::Tensor<T, 6>& christoffel_tilde,
						const tensorium::Tensor<T, 5>& dGamma_contract,
						const tensorium::Tensor<T, 7>& d2gamma_tilde) {
					
					using tensorium::Tensor;

					const auto& shape = gamma_tilde.shape();
					const size_t NX = shape[0], NY = shape[1], NZ = shape[2];

					Tensor<T, 5> Ricci_tilde({NX, NY, NZ, 3, 3});
					Ricci_tilde.fill(T(0));

					for (size_t x = 1; x + 1 < NX; ++x) {
						for (size_t y = 1; y + 1 < NY; ++y) {
							for (size_t z = 1; z + 1 < NZ; ++z) {

								for (size_t i = 0; i < 3; ++i) {
									for (size_t j = 0; j < 3; ++j) {

										T term_laplacian = T(0);
										for (size_t k = 0; k < 3; ++k) {
											for (size_t l = 0; l < 3; ++l) {
												T ginv_kl = gamma_tilde_inv({x, y, z, k, l});
												T d2_kl = d2gamma_tilde({x, y, z, i, j, k, l});
												term_laplacian += ginv_kl * d2_kl;
											}
										}
										term_laplacian *= T(-0.5);

										T term_divGamma = T(0);
										for (size_t m = 0; m < 3; ++m) {
											T dG_j = dGamma_contract({x, y, z, m, j});
											T dG_i = dGamma_contract({x, y, z, m, i});
											T g_mi = gamma_tilde({x, y, z, m, i});
											T g_mj = gamma_tilde({x, y, z, m, j});
											term_divGamma += g_mi * dG_j + g_mj * dG_i;
										}

										T term_gamma_gamma = T(0);
										for (size_t m = 0; m < 3; ++m) {
											for (size_t l = 0; l < 3; ++l) {
												T g1 = christoffel_tilde({x, y, z, m, l, i});
												T g2 = christoffel_tilde({x, y, z, l, m, j});
												term_gamma_gamma += g1 * g2;
											}
										}

										Ricci_tilde({x, y, z, i, j}) = term_laplacian + term_divGamma + term_gamma_gamma;
									}
								}
							}
						}
					}

					return Ricci_tilde;
				}

		};

}
