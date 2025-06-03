/**
 * @file BSSN.hpp
 * @brief Core implementation of the BSSN (Baumgarte–Shapiro–Shibata–Nakamura) formalism for general relativity.
 *
 * This file defines the BSSN grid structure and the initialization logic required to populate BSSN variables from a given metric.
 * It computes the conformal metric, its derivatives, the conformal Christoffel symbols, the extrinsic curvature tensor \( K_{ij} \),
 * and the trace-free conformal extrinsic curvature tensor \( \tilde{A}_{ij} \).
 */

#pragma once

#include "../Metric.hpp"
#include "../../Core/Vector.hpp"
#include "../../Core/Tensor.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNMetricUtils.hpp"
#include "BSSNDerivatives.hpp"
#include "BSSNextrinTensor.hpp"
#include "BSSNAtildeTensor.hpp"

namespace tensorium_RG {
	/**
	 * @struct BSSNGrid
	 * @brief Storage structure for all evolved BSSN variables on a single grid point or patch.
	 *
	 * This struct collects the geometric and gauge variables required in the BSSN formulation:
	 * - Lapse function \f$\alpha\f$
	 * - Shift vector \f$\beta^i\f$
	 * - Physical metric \f$\gamma_{ij}\f$ and its inverse
	 * - Conformal factor \f$\chi\f$
	 * - Conformal metric \f$\tilde{\gamma}_{ij}\f$ and its inverse
	 * - Partial derivatives \f$\partial_k \tilde{\gamma}_{ij}\f$
	 * - Conformal Christoffel symbols \f$\tilde{\Gamma}^k_{ij}\f$
	 * - Extrinsic curvature \f$K_{ij}\f$
	 * - Trace-free conformal extrinsic curvature \f$\tilde{A}_{ij}\f$
	 */
	struct BSSNGrid {
		std::vector<double> alpha;                                   ///< Lapse function α
		std::vector<tensorium::Vector<double>> beta;                 ///< Shift vector β^i
		std::vector<tensorium::Tensor<double, 2>> gamma_ij;          ///< Physical 3-metric γ_ij
		std::vector<tensorium::Tensor<double, 2>> gamma_ij_inv;      ///< Inverse γ^ij
		std::vector<double> chi;                                     ///< Conformal factor χ
		std::vector<tensorium::Tensor<double, 2>> gamma_tilde;       ///< Conformal metric \tilde{γ}_ij
		std::vector<tensorium::Tensor<double, 2>> gamma_tilde_inv;   ///< Inverse conformal metric \tilde{γ}^ij
		std::vector<tensorium::Tensor<double, 3>> dgamma_tilde;      ///< Derivatives ∂_k \tilde{γ}_ij
		std::vector<tensorium::Tensor<double, 3>> christoffel_tilde; ///< Conformal Christoffel symbols \tilde{Γ}^k_{ij}
		std::vector<tensorium::Tensor<double, 2>> ExtrinsicTensor;   ///< Extrinsic curvature K_ij
		std::vector<tensorium::Tensor<double, 2>> A_tildeTensor;     ///< Trace-free conformal extrinsic curvature \tilde{A}_{ij}
	};


	template<typename T>
		class BSSN {
			public:
				BSSNGrid grid;

				void init_BSSN(const tensorium::Vector<T>& X,
						const tensorium_RG::Metric<T>& metric,
						T dx, T dy, T dz) {

					T alpha;
					tensorium::Vector<T> beta(3);
					tensorium::Tensor<T, 2> gammaj({3, 3});
					metric.BSSN(X, alpha, beta, gammaj);
					tensorium::Tensor<T, 3> christoffel({3, 3, 3});
					tensorium::Tensor<T, 2> gammaj_inv = inv_mat_tensor(gammaj);
					tensorium::Tensor<T, 2> dgt({3, 3}); 
					tensorium::Tensor<T, 2> partial_beta({3, 3}); 
					tensorium::Tensor<T, 2> d_beta({3, 3});
					T chi = metric.compute_conformal_factor(gammaj);
					tensorium::Tensor<T, 2> gamma_tilde = compute_conformal_metric(metric, gammaj, chi);
					tensorium::Tensor<T, 2> gamma_tilde_inv = inv_mat_tensor(gamma_tilde);
					tensorium::Tensor<T, 3> dgamma_tilde({3, 3, 3});
					tensorium_RG::ExtrinsicCurvature<T> extr;
					tensorium_RG::BSSNAtildeTensor<T> Aij;

					tensorium_RG::compute_partial_derivatives_tensor2D<T>(
							X, dx, dy, dz,
							[&](const tensorium::Vector<T>& Xs, tensorium::Tensor<T, 2>& out) {
							T a_tmp;
							tensorium::Vector<T> b_tmp(3);
							tensorium::Tensor<T, 2> g_tmp({3, 3});
							metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
							T chi_tmp = compute_conformal_factor(metric, g_tmp);
							out = compute_conformal_metric(metric, g_tmp, chi_tmp);
							},
							dgamma_tilde
							);

					compute_christoffel_3D(gamma_tilde, dgamma_tilde, gamma_tilde_inv, christoffel);

					tensorium_RG::compute_partial_derivatives_vector<T>(
							X, dx, dy, dz,
							[&](const tensorium::Vector<T>& Xs, tensorium::Vector<T>& out) {
							T a_tmp;
							tensorium::Vector<T> b_tmp(3);
							tensorium::Tensor<T, 2> g_tmp({3, 3});
							metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
							out = b_tmp;
							},
							d_beta
							);

					for (size_t i = 0; i < 3; ++i)
						for (size_t j = 0; j < 3; ++j) {
							T val = 0;
							for (size_t l = 0; l < 3; ++l)
								val += gammaj(j, l) * d_beta(i, l); 
							partial_beta(i, j) = val;
						}

					auto Kij = extr.compute_Kij(dgt, gammaj, beta, partial_beta, christoffel, alpha);
					auto AtildeTensor = Aij.compute_Atilde_tensor(Kij, gammaj_inv, gammaj, chi);

					grid.ExtrinsicTensor = {Kij};
					grid.A_tildeTensor = {AtildeTensor};
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
