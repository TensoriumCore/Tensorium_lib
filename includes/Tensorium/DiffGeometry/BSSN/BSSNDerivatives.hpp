#pragma once

#include "../Metric.hpp"

namespace tensorium_RG {
	template<typename T, typename GammaFunc>
		inline void compute_partial_derivatives_gamma_tilde(
				const tensorium::Vector<T>& X,
				T dx, T dy, T dz,
				GammaFunc&& gamma_tilde_func,
				tensorium::Tensor<T, 3>& dgamma_out
				) {
			dgamma_out.resize(3, 3, 3);

			auto shifted = [&](T dx_, T dy_, T dz_) {
				tensorium::Vector<T> Xs = X;
				Xs(0) += dx_; Xs(1) += dy_; Xs(2) += dz_;
				tensorium::Tensor<T, 2> out({3, 3});
				gamma_tilde_func(Xs, out);
				return out;
			};

			for (size_t a = 0; a < 3; ++a) {
				for (size_t b = 0; b < 3; ++b) {
					tensorium::Tensor<T, 2> gm2 = shifted(-2*dx, 0, 0);
					tensorium::Tensor<T, 2> gm1 = shifted(-dx, 0, 0);
					tensorium::Tensor<T, 2> gp1 = shifted(dx, 0, 0);
					tensorium::Tensor<T, 2> gp2 = shifted(2*dx, 0, 0);
					dgamma_out(0, a, b) = (-gp2(a,b) + 8*gp1(a,b) - 8*gm1(a,b) + gm2(a,b)) / (12 * dx);

					gm2 = shifted(0, -2*dy, 0);
					gm1 = shifted(0, -dy, 0);
					gp1 = shifted(0, dy, 0);
					gp2 = shifted(0, 2*dy, 0);
					dgamma_out(1, a, b) = (-gp2(a,b) + 8*gp1(a,b) - 8*gm1(a,b) + gm2(a,b)) / (12 * dy);

					gm2 = shifted(0, 0, -2*dz);
					gm1 = shifted(0, 0, -dz);
					gp1 = shifted(0, 0, dz);
					gp2 = shifted(0, 0, 2*dz);
					dgamma_out(2, a, b) = (-gp2(a,b) + 8*gp1(a,b) - 8*gm1(a,b) + gm2(a,b)) / (12 * dz);
				}
			}
		}
}

