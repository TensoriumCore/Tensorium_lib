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

	template<typename T, typename TensorFunc>
		void compute_second_derivatives_tensor2D(
				const tensorium::Vector<T>& X,
				T dx, T dy, T dz,
				TensorFunc&& func,
				tensorium::Tensor<T, 4>& out
				) {
			out = tensorium::Tensor<T, 4>({3, 3, 3, 3});

			auto shifted = [&](T dx_, T dy_, T dz_) {
				tensorium::Vector<T> Xs = X;
				Xs(0) += dx_; Xs(1) += dy_; Xs(2) += dz_;
				tensorium::Tensor<T, 2> out_tensor({3, 3});
				func(Xs, out_tensor);
				return out_tensor;
			};

			for (size_t a = 0; a < 3; ++a) {
				for (size_t b = 0; b < 3; ++b) {
					// ∂²/∂x²
					auto gm2 = shifted(-2 * dx, 0, 0);
					auto gm1 = shifted(-dx, 0, 0);
					auto g0  = shifted(0, 0, 0);
					auto gp1 = shifted(dx, 0, 0);
					auto gp2 = shifted(2 * dx, 0, 0);
					out(0, a, b, 0) = (-gp2(a,b) + 16*gp1(a,b) - 30*g0(a,b) + 16*gm1(a,b) - gm2(a,b)) / (12 * dx * dx);

					// ∂²/∂y²
					gm2 = shifted(0, -2 * dy, 0);
					gm1 = shifted(0, -dy, 0);
					g0  = shifted(0, 0, 0);
					gp1 = shifted(0, dy, 0);
					gp2 = shifted(0, 2 * dy, 0);
					out(1, a, b, 1) = (-gp2(a,b) + 16*gp1(a,b) - 30*g0(a,b) + 16*gm1(a,b) - gm2(a,b)) / (12 * dy * dy);

					// ∂²/∂z²
					gm2 = shifted(0, 0, -2 * dz);
					gm1 = shifted(0, 0, -dz);
					g0  = shifted(0, 0, 0);
					gp1 = shifted(0, 0, dz);
					gp2 = shifted(0, 0, 2 * dz);
					out(2, a, b, 2) = (-gp2(a,b) + 16*gp1(a,b) - 30*g0(a,b) + 16*gm1(a,b) - gm2(a,b)) / (12 * dz * dz);
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

	template <typename T>
		tensorium::Tensor<T, 2> compute_dt_gamma_from_beta(
				const tensorium::Tensor<T, 2>& gamma,
				const tensorium::Vector<T>& beta,
				const tensorium::Tensor<T, 2>& partial_beta,
				const tensorium::Tensor<T, 3>& christoffel) {

			tensorium::Tensor<T, 2> dtg({3, 3});
			for (size_t i = 0; i < 3; ++i)
				for (size_t j = 0; j < 3; ++j) {
					T val = partial_beta(i, j) + partial_beta(j, i);
					for (size_t k = 0; k < 3; ++k)
						val -= 2.0 * christoffel(i, j, k) * beta(k);
					dtg(i, j) = val;
				}
			return dtg;
		}

	template<typename T>
		inline void compute_partial_derivatives_vector3D(
				const tensorium::Tensor<T, 4>& vec_field,
				size_t i, size_t j, size_t k,
				T dx, T dy, T dz,
				tensorium::Tensor<T, 2>& dvec_out 
				) {
			dvec_out.resize(3, 3); 

			auto get = [&](int ii, int jj, int kk, int a) -> T {
				return vec_field(ii, jj, kk, a);
			};

			const auto& shape = vec_field.shape();
			const size_t NX = shape[0], NY = shape[1], NZ = shape[2];

			for (size_t a = 0; a < 3; ++a) {
				if (i >= 2 && i + 2 < NX)
					dvec_out(0, a) = (-get(i+2,j,k,a) + 8*get(i+1,j,k,a) - 8*get(i-1,j,k,a) + get(i-2,j,k,a)) / (12 * dx);
				else if (i > 0 && i + 1 < NX)
					dvec_out(0, a) = (get(i+1,j,k,a) - get(i-1,j,k,a)) / (2 * dx);
				else
					dvec_out(0, a) = T(0);

				if (j >= 2 && j + 2 < NY)
					dvec_out(1, a) = (-get(i,j+2,k,a) + 8*get(i,j+1,k,a) - 8*get(i,j-1,k,a) + get(i,j-2,k,a)) / (12 * dy);
				else if (j > 0 && j + 1 < NY)
					dvec_out(1, a) = (get(i,j+1,k,a) - get(i,j-1,k,a)) / (2 * dy);
				else
					dvec_out(1, a) = T(0);

				if (k >= 2 && k + 2 < NZ)
					dvec_out(2, a) = (-get(i,j,k+2,a) + 8*get(i,j,k+1,a) - 8*get(i,j,k-1,a) + get(i,j,k-2,a)) / (12 * dz);
				else if (k > 0 && k + 1 < NZ)
					dvec_out(2, a) = (get(i,j,k+1,a) - get(i,j,k-1,a)) / (2 * dz);
				else
					dvec_out(2, a) = T(0);
			}
		}

}

