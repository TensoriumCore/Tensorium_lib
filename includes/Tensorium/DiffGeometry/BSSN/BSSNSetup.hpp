#pragma once

#include "../Metric.hpp"
#include "../../Core/Vector.hpp"
#include "../../Core/Tensor.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNMetricUtils.hpp"
#include "BSSNDerivatives.hpp"
namespace tensorium {

	struct BSSNGrid {
		std::vector<double> alpha;
		std::vector<Vector<double>> beta;
		std::vector<Tensor<double, 2>> gamma_ij;
		std::vector<Tensor<double, 2>> gamma_ij_inv;

		std::vector<double> chi;
		std::vector<Tensor<double, 2>> gamma_tilde;
		std::vector<Tensor<double, 2>> gamma_tilde_inv;

		std::vector<Tensor<double, 3>> dgamma_tilde;
		std::vector<Tensor<double, 3>> christoffel_tilde;

	};


	template<typename T>
		class BSSN {
			public:
				BSSNGrid grid;

				void init_BSSN(const Vector<T>& X,
						const tensorium_RG::Metric<T>& metric,
						T dx, T dy, T dz) {

					T alpha;
					Vector<T> beta(3);
					Tensor<T, 2> gammaj({3, 3});
					metric.BSSN(X, alpha, beta, gammaj);

					Tensor<T, 2> gammaj_inv = inv_mat_tensor(gammaj);

					T chi = metric.compute_conformal_factor(gammaj);
					Tensor<T, 2> gamma_tilde = compute_conformal_metric(metric, gammaj, chi);
					Tensor<T, 2> gamma_tilde_inv = inv_mat_tensor(gamma_tilde);
					std::cout<<"\n--- Spatial metric γ_{ij} ---\n";
					gamma_tilde.print();

					std::cout<<"\n--- Spatial metric γ^{ij} ---\n";
					gamma_tilde_inv.print();
					Tensor<T, 3> dgamma_tilde({3, 3, 3});
					compute_partial_derivatives_gamma_tilde(
							X, dx, dy, dz,
							[&](const Vector<T>& Xs, Tensor<T, 2>& out) {
							T a_tmp;
							Vector<T> b_tmp(3);
							Tensor<T, 2> g_tmp({3, 3});
							metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
							T chi_tmp = compute_conformal_factor(metric, g_tmp);
							out = compute_conformal_metric(metric, g_tmp, chi_tmp);
							},
							dgamma_tilde
							);

					std::cout<<"\n--- Partial derivatives Spatial metric γ^{ij} ---\n";
					dgamma_tilde.print();

					Tensor<T, 3> christoffel({3, 3, 3});
					compute_christoffel_3D(gamma_tilde, dgamma_tilde, gamma_tilde_inv, christoffel);
					std::cout << "\n--- 3D christoffel symbols\n";
					christoffel.print();
					grid.alpha = {alpha};
					grid.beta = {beta};
					grid.gamma_ij = {gammaj};
					grid.gamma_ij_inv = {gammaj_inv};
					grid.chi = {chi};
					grid.gamma_tilde = {gamma_tilde};
					grid.gamma_tilde_inv = {gamma_tilde_inv};
					grid.dgamma_tilde = {dgamma_tilde};
					grid.christoffel_tilde = {christoffel};
				}
		};

} // namespace tensorium
