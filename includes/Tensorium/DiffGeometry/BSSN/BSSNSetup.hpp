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
#include "BSSNAutoDiff.hpp"
#include "BSSNextrinTensor.hpp"
#include "BSSNAtildeTensor.hpp"
#include "BSSNTildeChristoffel.hpp"
#include "BSSNContractedChristoffel.hpp"
#include "BSSNPrintDebug.hpp"
#include "BSSNConstraintsSolver.hpp"
#include "BSSNRicciTensor.hpp"
#include <iostream>

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
	struct alignas(32) BSSNGrid {

		std::vector<double> alpha;                                   ///< Lapse function \f$\alpha\f$
		std::vector<tensorium::Vector<double>> beta;                 ///< Shift vector \f$\beta^i\f$
		std::vector<tensorium::Tensor<double, 2>> gamma_ij;          ///< Physical 3-metric \f$\gamma_{ij}\f$
		std::vector<tensorium::Tensor<double, 2>> gamma_ij_inv;      ///< Inverse \f$\gamma^{ij}\f$
		std::vector<double> chi;                                     ///< Conformal factor \f$\chi\f$
		std::vector<tensorium::Tensor<double, 2>> gamma_tilde;       ///< Conformal metric \f$\tilde{\gamma}_{ij}\f$
		std::vector<tensorium::Tensor<double, 2>> gamma_tilde_inv;   ///< Inverse \f$\tilde{\gamma}^{ij}\f$
		std::vector<tensorium::Tensor<double, 3>> dgamma_tilde;      ///< Derivatives \f$\partial_k \tilde{\gamma}_{ij}\f$
		std::vector<tensorium::Tensor<double, 3>> christoffel_tilde; ///< Christoffel symbols \f$\tilde{\Gamma}^k_{ij}\f$
		std::vector<tensorium::Tensor<double, 2>> ExtrinsicTensor;   ///< Extrinsic curvature \f$K_{ij}\f$
		std::vector<tensorium::Tensor<double, 2>> A_tildeTensor;     ///< Trace-free extrinsic curvature \f$\tilde{A}_{ij}\f$
		std::vector<tensorium::Vector<double>> tilde_Gamma;			 ///< Conformal contracted symbols \f$\tilde{\Gamma}^i\f$
		std::vector<tensorium::Vector<double>> contracted_Gamma;	 ///< Contracted symbols \f$\Gamma^i_{ij} = -\frac{3}{2} \partial_j \ln \chi\f$
		std::vector<tensorium::Tensor<double, 5>> ricci_tilde;		// [NX, NY, NZ, 3, 3]

	};
	/**
	 * @class BSSN
	 * @brief Driver class to initialize and store BSSN variables from an input spacetime metric.
	 *
	 * This class initializes all core variables required for BSSN evolution using a given spacetime metric.
	 * The metric must provide access to the lapse, shift, and spatial metric via `metric.BSSN(X, α, β^i, γ_ij)`.
	 *
	 * @tparam T Numeric type (e.g. double)
	 */
	template<typename T>
		class BSSN {
			public:
				BSSNGrid grid;

				void init_BSSN(const tensorium::Vector<T>& X,
						const tensorium_RG::Metric<T>& metric,
						T dx, T dy, T dz) {
					T alpha;
					tensorium::Vector<T> beta(3);
					tensorium::Tensor<T,2> gamma_ij({3, 3});
					metric.BSSN(X, alpha, beta, gamma_ij);

					tensorium::Tensor<T,2> dgt({3, 3});
					tensorium::Tensor<T,2> gamma_ij_inv = inv_mat_tensor(gamma_ij);
					T chi = metric.compute_conformal_factor(gamma_ij);
					tensorium::Tensor<T,2> gamma_tilde = compute_conformal_metric(metric, gamma_ij, chi);
					tensorium::Tensor<T,2> gamma_tilde_inv = inv_mat_tensor(gamma_tilde);

					auto dgamma_tilde = autodiff(X, dx, dy, dz,
							[&](const tensorium::Vector<T>& Xs) {
								T a_tmp;
								tensorium::Vector<T> b_tmp(3);
								tensorium::Tensor<T,2> g_tmp({3, 3});
								metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
								T chi_tmp = compute_conformal_factor(metric, g_tmp);
								return compute_conformal_metric(metric, g_tmp, chi_tmp);
							},
							DiffMode::PARTIAL
							);

					tensorium::Tensor<T,3> christoffel_tilde({3, 3, 3});
					compute_christoffel_3D(gamma_tilde, dgamma_tilde, gamma_tilde_inv, christoffel_tilde);


					tensorium::Vector<T> tilde_Gamma(3);
					TildeGamma<T>::compute(gamma_tilde_inv, christoffel_tilde, tilde_Gamma);

					auto contracted_Gamma = tensorium_RG::BSSNContractedGamma<T>::compute(
							X, metric, dx, dy, dz, chi);

					auto d_beta = autodiff(X, dx, dy, dz,
							[&](const tensorium::Vector<T>& Xs) {
								T a_tmp;
								tensorium::Vector<T> b_tmp(3);
								tensorium::Tensor<T,2> g_tmp({3, 3});
								metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
								return b_tmp;
							},
							DiffMode::PARTIAL
							);

					auto dgamma_phys = autodiff(X, dx, dy, dz,
							[&](const tensorium::Vector<T>& Xs) {
								T a_tmp;
								tensorium::Vector<T> b_tmp(3);
								tensorium::Tensor<T,2> g_tmp({3, 3});
								metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
								return g_tmp;
							},
							DiffMode::PARTIAL
							);

					tensorium::Tensor<T,3> christoffel_phys({3, 3, 3});
					compute_christoffel_3D(gamma_ij, dgamma_phys, gamma_ij_inv, christoffel_phys);

					tensorium_RG::ExtrinsicCurvature<T> extr;
					auto Kij = extr.compute_Kij(dgt, gamma_ij, beta, d_beta, christoffel_phys, alpha);

					tensorium_RG::BSSNAtildeTensor<T> Aij;
					auto AtildeTensor = Aij.compute_Atilde_tensor(Kij, gamma_ij_inv, gamma_ij, chi);

					const size_t NX = 1, NY = 1, NZ = 1;

					tensorium::Tensor<T,5> Atilde_full({NX, NY, NZ, 3, 3});
					tensorium::Tensor<T,5> gtilde_inv_full({NX, NY, NZ, 3, 3});
					for (size_t i = 0; i < NX; ++i) {
						for (size_t j = 0; j < NY; ++j) {
							for (size_t k = 0; k < NZ; ++k) {
								for (int a = 0; a < 3; ++a) {
									for (int b = 0; b < 3; ++b) {
										Atilde_full(std::array<size_t,5>{i, j, k, (size_t)a, (size_t)b}) = AtildeTensor(a, b);
										gtilde_inv_full(std::array<size_t,5>{i, j, k, (size_t)a, (size_t)b}) = gamma_tilde_inv(a, b);
									}
								}
							}
						}
					}

					auto psi_full = ConstraintSolver<T>::solveLichnerowicz(
							Atilde_full,
							gtilde_inv_full,
							dx, dy, dz,
							2000,
							T(1e-8)
							);
					{
						T psi_000 = std::fmax(psi_full(std::array<size_t,3>{0, 0, 0}), T(1e-8));
						T new_chi = T(1) / std::pow(psi_000, T(4));
						grid.chi = { new_chi };
					}

					auto R1 = RicciTensor3D<double>::compute_laplacian_term(X, dx, dy, dz, metric, gamma_tilde_inv);
					auto R2 = RicciTensor3D<double>::compute_dGamma_term(X, dx, dy, dz, tilde_Gamma, gamma_tilde);
					auto R3 = RicciTensor3D<double>::compute_GammaGamma_term(tilde_Gamma, christoffel_tilde, gamma_tilde);
					auto R4 = RicciTensor3D<T>::compute_GammaProduct_term(gamma_tilde_inv, christoffel_tilde);
					tensorium::Tensor<T,2> Ricci({3,3});
					for (size_t i = 0; i < 3; ++i) {
						for (size_t j = 0; j < 3; ++j) {
							Ricci(i,j) = R1(i,j)
								+ R2(i,j)
								+ R3(i,j)
								+ R4(i,j);
						}
					}
					

					grid.alpha             = {alpha};
					grid.beta              = {beta};
					grid.gamma_ij          = {gamma_ij};
					grid.gamma_ij_inv      = {gamma_ij_inv};
					grid.chi               = {chi};
					grid.gamma_tilde       = {gamma_tilde};
					grid.gamma_tilde_inv   = {gamma_tilde_inv};
					grid.dgamma_tilde      = {dgamma_tilde};
					grid.christoffel_tilde = {christoffel_tilde};
					grid.tilde_Gamma       = {tilde_Gamma};
					grid.contracted_Gamma  = {contracted_Gamma};
					grid.ExtrinsicTensor   = {Kij};
					grid.A_tildeTensor     = {AtildeTensor};


					std::cout << std::setprecision(6) << std::fixed;
					std::cout << "\n========= BSSN Quantities at X =========\n";
					
					std::cout << "--- alpha ---\n" << grid.alpha[0] << "\n";
					print_vector("beta", grid.beta[0]);
					
					std::cout << "dbeta\n";
					d_beta.print();
					
					print_tensor2("gamma_ij", grid.gamma_ij[0]);
					print_tensor2("gamma_ij_inv", grid.gamma_ij_inv[0]);
					
					print_tensor2("gamma_tilde", grid.gamma_tilde[0]);
					print_tensor2("gamma_tilde_inv", grid.gamma_tilde_inv[0]);
					
					print_tensor3("dgamma_tilde", grid.dgamma_tilde[0]);
					print_tensor3("christoffel_tilde", grid.christoffel_tilde[0]);
					print_vector("tilde_Gamma", grid.tilde_Gamma[0]);
					
					print_tensor2("∂_t gamma_ij (dgt)", dgt);
					print_vector("contracted_Gamma", grid.contracted_Gamma[0]);
					
					print_tensor3("dgamma_phys", dgamma_phys);
					print_tensor3("christoffel_phys", christoffel_phys);
					print_tensor2("Extrinsic curvature K_ij", grid.ExtrinsicTensor[0]);
					print_tensor2("A_tilde_ij", grid.A_tildeTensor[0]);
					std::cout << "--- chi ---\n" << grid.chi[0] << "\n";
					
					print_tensor2("R1 = -1/2 ∇² γ̃_ij", R1);
					print_tensor2("R2 = 1/2 (∂_j Γ̃^k γ̃_{ki} + ∂_i Γ̃^k γ̃_{kj})", R2);
					print_tensor2("R3 = Γ̃^k Γ̃_{(i j) k}", R3);
					std::cout << "\n--- R4 = γ̃^{ℓm} [ 2 Γ̃^k_{ℓ(i} Γ̃_{j) k m} + Γ̃^k_{i m} Γ̃_{k ℓ j} ] ---\n";
					print_tensor2("R4", R4);
					print_tensor2("Full Ricci  R_ij", Ricci);
					std::cout << "========================================\n\n";
					
					
				}
		};
}
