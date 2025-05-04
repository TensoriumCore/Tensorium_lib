#pragma once

#include "../Core/Vector.hpp"
#include "../Core/Tensor.hpp"
#include "../DiffGeometry/ChristoffelSymbol.hpp"
#include "../DiffGeometry/Metric.hpp"
#include "../DiffGeometry/RiemannTensor.hpp"

namespace morpheus {


		template <typename T>
			morpheus::Tensor<T, 2> inv_mat_tensor(const morpheus::Tensor<T, 2>& g) {
				Matrix<T> mat = tensor_to_matrix(g);
				Matrix<T> inv = mat.inverse();
				return matrix_to_tensor(inv);
			}

		template <typename T>
			Matrix<T> tensor_to_matrix(const Tensor<T, 2>& tensor) {
				const size_t d0 = tensor.dimensions[0];
				const size_t d1 = tensor.dimensions[1];
				Matrix<T> mat({d0, d1});
				for (size_t i = 0; i < d0; ++i)
					for (size_t j = 0; j < d1; ++j)
						mat(i, j) = tensor(i, j);
				return mat;
			}

		template <typename T>
			Tensor<T, 2> matrix_to_tensor(const Matrix<T>& mat) {
				const size_t d0 = mat.rows;
				const size_t d1 = mat.cols;

				Tensor<T, 2> tensor({d0, d1});
				for (size_t i = 0; i < d0; ++i)
					for (size_t j = 0; j < d1; ++j)
						tensor(i, j) = mat(i, j);
				return tensor;
			}

		/*
		 * @brief Compute Christoffel symbols Γ^λ_{μν}
		 * @param X 4-position
		 * @param h Finite difference step
		 * @param g Metric tensor g_{μν}
		 * @param g_inv Inverse metric g^{μν}
		 * @param metric_generator Callable: X ↦ g_{μν}
		 * @return Christoffel symbol tensor Γ^λ_{μν}
		 */
		
		template <typename T, typename MetricFunc>
			morpheus_RG::ChristoffelSym<T> compute_christoffel(
					const morpheus::Vector<T>& X, T h,
					const morpheus::Tensor<T, 2>& g,
					const morpheus::Tensor<T, 2>& g_inv,
					MetricFunc&& metric_generator)
			{
				return morpheus_RG::ChristoffelSym<T>::compute_christoffel(X, h, g, g_inv, metric_generator);
			}

		/*
		 * @brief Generate a metric tensor g_{μν} at position X using a Metric<T> object
		 * @param metric Metric object (Minkowski, Schwarzschild, Kerr)
		 * @param X Point at which to evaluate
		 * @param g Output tensor g_{μν}
		 */
		template <typename T>
			void generate_metric(const morpheus_RG::Metric<T>& metric,
					const Vector<T>& X,
					Tensor<T, 2>& g) {
				metric(X, g);
			}

		template <typename T>
			inline morpheus::Tensor<T, 4> compute_riemann_tensor(
					const morpheus::Vector<T>& X,
					T h,
					const morpheus_RG::Metric<T>& metric)
			{
				return morpheus_RG::RiemannTensor<T>::compute(X, h, metric);
			}

		template <typename T>
			inline void print_riemann_tensor(const morpheus::Tensor<T, 4>& R)
			{
				morpheus_RG::RiemannTensor<T>::print_componentwise(R);
			}


		template<typename T>
			inline Tensor<T, 2> contract_riemann_to_ricci(const Tensor<T, 4>& R, const Tensor<T, 2>& ginv) {
				return morpheus_RG::RiemannTensor<T>::contract_to_ricci(R, ginv);
			}

}
