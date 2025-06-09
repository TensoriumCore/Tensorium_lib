#pragma once

#include "../Core/Vector.hpp"
#include "../Core/Tensor.hpp"
#include "../DiffGeometry/ChristoffelSymbol.hpp"
#include "../DiffGeometry/Metric.hpp"
#include "../DiffGeometry/RiemannTensor.hpp"
#include "../DiffGeometry/RicciTensor.hpp"
#include "../DiffGeometry/BSSN/BSSNChristoffel.hpp"
#include "../DiffGeometry/BSSN/BSSNSetup.hpp"
#include "../DiffGeometry/BSSN/BSSNMetricUtils.hpp"

namespace tensorium {


		template <typename T>
			tensorium::Tensor<T, 2> inv_mat_tensor(const tensorium::Tensor<T, 2>& g) {
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
			tensorium_RG::ChristoffelSym<T> compute_christoffel(
					const tensorium::Vector<T>& X, T h,
					const tensorium::Tensor<T, 2>& g,
					const tensorium::Tensor<T, 2>& g_inv,
					MetricFunc&& metric_generator)
			{
				return tensorium_RG::ChristoffelSym<T>::compute_christoffel(X, h, g, g_inv, metric_generator);
			}

		/*
		 * @brief Generate a metric tensor g_{μν} at position X using a Metric<T> object
		 * @param metric Metric object (Minkowski, Schwarzschild, Kerr)
		 * @param X Point at which to evaluate
		 * @param g Output tensor g_{μν}
		 */
		template <typename T>
			void generate_metric(const tensorium_RG::Metric<T>& metric,
					const Vector<T>& X,
					Tensor<T, 2>& g) {
				metric(X, g);
			}

		template <typename T>
			inline tensorium::Tensor<T, 4> compute_riemann_tensor(
					const tensorium::Vector<T>& X,
					T h,
					const tensorium_RG::Metric<T>& metric)
			{
				return tensorium_RG::RiemannTensor<T>::compute(X, h, metric);
			}

		template <typename T>
			inline void print_riemann_tensor(const tensorium::Tensor<T, 4>& R)
			{
				tensorium_RG::RiemannTensor<T>::print_componentwise(R);
			}


		template<typename T>
			inline Tensor<T, 2> contract_riemann_to_ricci(const Tensor<T, 4>& R, const Tensor<T, 2>& ginv) {
				return tensorium_RG::RicciTensor<T>::contract_to_ricci(R, ginv);
			}
		
		template<typename T>
			inline T compute_ricci_scalar(const Tensor<T, 2>& Ricci, const Tensor<T, 2>& ginv) {
				return tensorium_RG::RicciTensor<T>::compute_ricci_scalar(Ricci, ginv);
			}

		template<typename T>
			inline void print_ricci_tensor(const Tensor<T, 2>& R) {
				tensorium_RG::RicciTensor<T>::print_componentwise(R);
			}

		template<typename T>
			inline void print_ricci_scalar(const Tensor<T, 2>& Ricci, const Tensor<T, 2>& g_inv) {
				tensorium_RG::RicciTensor<T>::print_ricci_scalar(Ricci, g_inv);
			}
		template<typename T>
			inline T compute_conformal_factor(const tensorium_RG::Metric<T>& metric,
					const Tensor<T, 2>& gamma) {
				return metric.compute_conformal_factor(gamma);
			}

		template<typename T>
			inline Tensor<T, 2> compute_conformal_metric(const tensorium_RG::Metric<T>& metric,
					const Tensor<T, 2>& gamma,
					T chi) {
				Tensor<T, 2> gamma_tilde({3, 3});
				metric.compute_conformal_metric(gamma, chi, gamma_tilde);
				return gamma_tilde;
			}

	
		template<typename T>
			inline void compute_christoffel_3D(const Tensor<T, 2>& gamma_tilde,
					const Tensor<T, 3>& dgamma_tilde,
					const Tensor<T, 2>& gamma_tilde_inv,
					Tensor<T, 3>& Christoffel) {
				tensorium_RG::BSSNChristoffel<T>::compute(gamma_tilde, dgamma_tilde, gamma_tilde_inv, Christoffel);
			}

		template<typename T>
			inline void compute_christoffel_3D_Grid(const Tensor<T, 5>& gamma_tilde,
					const Tensor<T, 6>& dgamma_tilde,
					const Tensor<T, 5>& gamma_tilde_inv,
					Tensor<T, 6>& Christoffel) {
				tensorium_RG::BSSNChristoffel<T>::compute3D(gamma_tilde, dgamma_tilde, gamma_tilde_inv, Christoffel);
			}

		template<typename T>
			inline tensorium_RG::BSSN<T> setup_BSSN(const Vector<T>& X,
					const tensorium_RG::Metric<T>& metric,
					T dx, T dy, T dz) {
				tensorium_RG::BSSN<T> bssn;
				bssn.init_BSSN(X, metric, dx, dy, dz);
				return bssn;
			}

}
