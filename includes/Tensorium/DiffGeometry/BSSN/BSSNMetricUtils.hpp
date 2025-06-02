#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include <cassert>
#include <cstddef>

namespace tensorium::bssn {

	template<typename T>
		Tensor<T, 5> generate_conformal_metric_field(
				const tensorium_RG::Metric<T>& metric,
				size_t Nx, size_t Ny, size_t Nz,
				T dx, T dy, T dz
				) {
			Tensor<T, 5> gamma_tilde_field({Nx, Ny, Nz, 3, 3});

			tensorium::Vector<T> X(4); // X^μ
			X(0) = T(0); 

			for (size_t i = 0; i < Nx; ++i) {
				T x = i * dx;
				for (size_t j = 0; j < Ny; ++j) {
					T y = j * dy;
					for (size_t k = 0; k < Nz; ++k) {
						T z = k * dz;

						X(1) = x;
						X(2) = y;
						X(3) = z;

						tensorium::Tensor<T, 2> gamma(3, 3), gamma_tilde(3, 3);
						tensorium::Vector<T> beta(3);
						T alpha, chi;

						metric.BSSN(X, alpha, beta, gamma);
						chi = metric.compute_conformal_factor(gamma);
						metric.compute_conformal_metric(gamma, chi, gamma_tilde);

						for (size_t a = 0; a < 3; ++a)
							for (size_t b = 0; b < 3; ++b)
								gamma_tilde_field(i, j, k, a, b) = gamma_tilde(a, b);
					}
				}
			}

			return gamma_tilde_field;
		}

}
