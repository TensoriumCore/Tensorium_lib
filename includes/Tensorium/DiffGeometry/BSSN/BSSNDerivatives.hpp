#pragma once

#include "../Metric.hpp"


namespace tensorium_RG {

	template<typename T, typename TensorFunc>
		void compute_partial_derivatives_tensor2D(
				const tensorium::Vector<T>& X,
				T dx, T dy, T dz,
				TensorFunc&& func,
				tensorium::Tensor<T, 3>& out 
				) {
			out.resize(3, 3, 3);

			auto shifted = [&](T dx_, T dy_, T dz_) {
				tensorium::Vector<T> Xs = X;
				Xs(0) += dx_; Xs(1) += dy_; Xs(2) += dz_;
				tensorium::Tensor<T, 2> out_tensor({3, 3});
				func(Xs, out_tensor);
				return out_tensor;
			};

			for (size_t a = 0; a < 3; ++a) {
				for (size_t b = 0; b < 3; ++b) {
					// ∂₀ = ∂/∂x
					auto gm2 = shifted(-2 * dx, 0, 0);
					auto gm1 = shifted(-dx, 0, 0);
					auto gp1 = shifted(dx, 0, 0);
					auto gp2 = shifted(2 * dx, 0, 0);
					out(0, a, b) = (-gp2(a,b) + 8*gp1(a,b) - 8*gm1(a,b) + gm2(a,b)) / (12 * dx);

					// ∂₁ = ∂/∂y
					gm2 = shifted(0, -2 * dy, 0);
					gm1 = shifted(0, -dy, 0);
					gp1 = shifted(0, dy, 0);
					gp2 = shifted(0, 2 * dy, 0);
					out(1, a, b) = (-gp2(a,b) + 8*gp1(a,b) - 8*gm1(a,b) + gm2(a,b)) / (12 * dy);

					// ∂₂ = ∂/∂z
					gm2 = shifted(0, 0, -2 * dz);
					gm1 = shifted(0, 0, -dz);
					gp1 = shifted(0, 0, dz);
					gp2 = shifted(0, 0, 2 * dz);
					out(2, a, b) = (-gp2(a,b) + 8*gp1(a,b) - 8*gm1(a,b) + gm2(a,b)) / (12 * dz);
				}
			}
		}

	template<typename T, typename VectorFunc>
		void compute_partial_derivatives_vector(
				const tensorium::Vector<T>& X,
				T dx, T dy, T dz,
				VectorFunc&& func,
				tensorium::Tensor<T, 2>& out 
				) {
			out.resize(3, 3);

			auto shifted = [&](T dx_, T dy_, T dz_) {
				tensorium::Vector<T> Xs = X;
				Xs(0) += dx_; Xs(1) += dy_; Xs(2) += dz_;
				tensorium::Vector<T> Vout(3);
				func(Xs, Vout);
				return Vout;
			};

			for (size_t i = 0; i < 3; ++i) {
				auto gm2 = shifted(-2*dx, 0, 0);
				auto gm1 = shifted(-dx, 0, 0);
				auto gp1 = shifted(dx, 0, 0);
				auto gp2 = shifted(2*dx, 0, 0);
				out(0, i) = (-gp2(i) + 8*gp1(i) - 8*gm1(i) + gm2(i)) / (12 * dx);

				gm2 = shifted(0, -2*dy, 0);
				gm1 = shifted(0, -dy, 0);
				gp1 = shifted(0, dy, 0);
				gp2 = shifted(0, 2*dy, 0);
				out(1, i) = (-gp2(i) + 8*gp1(i) - 8*gm1(i) + gm2(i)) / (12 * dy);

				gm2 = shifted(0, 0, -2*dz);
				gm1 = shifted(0, 0, -dz);
				gp1 = shifted(0, 0, dz);
				gp2 = shifted(0, 0, 2*dz);
				out(2, i) = (-gp2(i) + 8*gp1(i) - 8*gm1(i) + gm2(i)) / (12 * dz);
			}
		}

	template<typename T, typename ScalarFunc>
		void compute_partial_derivatives_scalar(
				const tensorium::Vector<T>& X,
				T dx, T dy, T dz,
				ScalarFunc&& func,
				tensorium::Vector<T>& out) {

			out.resize(3);

			auto shifted = [&](T dx_, T dy_, T dz_) {
				tensorium::Vector<T> Xs = X;
				Xs(0) += dx_; Xs(1) += dy_; Xs(2) += dz_;
				return func(Xs);
			};

			// ∂/∂x
			T gm2 = shifted(-2 * dx, 0, 0);
			T gm1 = shifted(-dx, 0, 0);
			T gp1 = shifted(dx, 0, 0);
			T gp2 = shifted(2 * dx, 0, 0);
			out[0] = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * dx);

			// ∂/∂y
			gm2 = shifted(0, -2 * dy, 0);
			gm1 = shifted(0, -dy, 0);
			gp1 = shifted(0, dy, 0);
			gp2 = shifted(0, 2 * dy, 0);
			out[1] = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * dy);

			// ∂/∂z
			gm2 = shifted(0, 0, -2 * dz);
			gm1 = shifted(0, 0, -dz);
			gp1 = shifted(0, 0, dz);
			gp2 = shifted(0, 0, 2 * dz);
			out[2] = (-gp2 + 8 * gp1 - 8 * gm1 + gm2) / (12 * dz);
		}
}

