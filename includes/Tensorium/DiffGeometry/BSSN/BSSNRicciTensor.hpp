#pragma once

#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Vector.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNContractedChristoffel.hpp"
#include "BSSNAutoDiff.hpp"

namespace tensorium_RG {

	template<typename T>
		class RicciTensor3D {
			public:
				/**
				 * @brief Compute R_ij^(1) = -½ γ̃^{kl} ∂ₖ∂ₗ γ̃_{ij} using autodiff scalar
				 *
				 * @param X         : Position vector
				 * @param dx,dy,dz  : Grid spacing
				 * @param metric    : Metric object (for γ̃_ij construction)
				 * @param gamma_tilde_inv : Inverse conformal metric at X
				 * @return tensorium::Tensor<T, 2> : Laplacian term of Ricci
				 */
				static tensorium::Tensor<T, 2> compute_laplacian_term(
						const tensorium::Vector<T>& X,
						T dx, T dy, T dz,
						const Metric<T>& metric,
						const tensorium::Tensor<T, 2>& gamma_tilde_inv)
				{
					using namespace tensorium;

					Tensor<T, 2> result({3, 3});

					for (size_t i = 0; i < 3; ++i) {
						for (size_t j = 0; j < 3; ++j) {

							auto tensor_func = [&](const Vector<T>& Xs, Tensor<T, 2>& out_tensor) {
								T a; Vector<T> b(3); Tensor<T, 2> g_phys({3, 3});
								metric.BSSN(Xs, a, b, g_phys);
								T chi = compute_conformal_factor(metric, g_phys);
								Tensor<T, 2> gtilde = compute_conformal_metric(metric, g_phys, chi);
								out_tensor(i, j) = gtilde(i, j);
							};

							Tensor<T, 4> d2_full({1, 1, 3, 3});
							compute_second_derivatives_tensor2D<T>(
									X, dx, dy, dz, tensor_func, d2_full);

							Tensor<T, 2> d2gamma({3, 3});
							for (int k = 0; k < 3; ++k)
								for (int l = 0; l < 3; ++l)
									d2gamma(k, l) = d2_full(0, 0, k, l);

							T laplace = 0.0;
							for (int k = 0; k < 3; ++k)
								for (int l = 0; l < 3; ++l)
									laplace += gamma_tilde_inv(k, l) * d2gamma(k, l);

							result(i, j) = -0.5 * laplace;
						}
					}

					for (size_t i = 0; i < 3; ++i) {
						for (size_t j = i + 1; j < 3; ++j) {
							T avg = 0.5 * (result(i, j) + result(j, i));
							result(i, j) = result(j, i) = avg;
						}
					}

					return result;
				}

				/**
				 * @brief Compute the term R^(2)_ij = (1/2)(∇_i Γ̃_j + ∇_j Γ̃_i) for the conformal Ricci tensor.
				 *
				 * This term is derived from the contraction of the conformal Christoffel symbols:
				 *   Γ̃^i = γ̃^{jk} Γ̃^i_{jk}
				 * and its spatial derivatives.
				 *
				 * @tparam T Scalar type (e.g., float, double)
				 * @param X           Spatial point
				 * @param dx, dy, dz  Grid spacing
				 * @param tilde_Gamma Contracted conformal Christoffel symbol at X (i.e. Γ̃^i)
				 * @return Symmetric 3×3 tensor representing (1/2)(∇_i Γ̃_j + ∇_j Γ̃_i)
				 */
				static tensorium::Tensor<T, 2> compute_dGamma_term(
						const tensorium::Vector<T>& X,
						T dx, T dy, T dz,
						const tensorium::Vector<T>& tilde_Gamma,
						const tensorium::Tensor<T, 2>& tilde_gamma_contract)
				{
					using namespace tensorium;

					Tensor<T,2> dGamma({3,3});
					for (int k = 0; k < 3; ++k) {
						auto scalar_func = [&](const Vector<T>& Xs) -> T {
							return tilde_Gamma(k);
						};
						Vector<T> grad_k = partial_scalar(X, dx, dy, dz, scalar_func);
						for (int j = 0; j < 3; ++j) {
							dGamma(j, k) = grad_k(j);
						}
					}

					Tensor<T,2> R2({3,3});
					for (int i = 0; i < 3; ++i) {
						for (int j = 0; j < 3; ++j) {
							T sum = 0;
							for (int k = 0; k < 3; ++k) {
								sum += tilde_gamma_contract(k,i) * dGamma(j,k)
									+ tilde_gamma_contract(k,j) * dGamma(i,k);
							}
							R2(i,j) = static_cast<T>(0.5) * sum;
						}
					}

					return R2;
				}
				/**
				 * @brief Compute R^{(3)}_{ij} = ½ Γ̃^k (γ̃_{k i} Γ̃^k_{j k} + γ̃_{k j} Γ̃^k_{i k})
				 *
				 * @param tilde_Gamma          Contracted conformal Christoffel Γ̃^k
				 * @param christoffel_tilde    Conformal Christoffel symbols Γ̃^i_{j k}
				 * @param gamma_tilde          Conformal metric γ̃_{ij}
				 * @return tensorium::Tensor<T,2>  The R^{(3)}_{ij} contribution
				 */
				static tensorium::Tensor<T,2> compute_GammaGamma_term(
						const tensorium::Vector<T>&      tilde_Gamma,
						const tensorium::Tensor<T,3>&    christoffel_tilde,
						const tensorium::Tensor<T,2>&    gamma_tilde)
				{
					using namespace tensorium;
					Tensor<T,2> R3({3, 3});

					for (size_t i = 0; i < 3; ++i) {
						for (size_t j = 0; j < 3; ++j) {
							T sum_ij = 0;
							for (size_t k = 0; k < 3; ++k) {
								T G_ijk = 0, G_jik = 0;
								for (size_t m = 0; m < 3; ++m) {
									G_ijk += gamma_tilde(i, m) * christoffel_tilde(m, j, k);
									G_jik += gamma_tilde(j, m) * christoffel_tilde(m, i, k);
								}
								sum_ij += tilde_Gamma(k) * (G_ijk + G_jik);
							}
							R3(i, j) = T(0.5) * sum_ij;
						}
					}

					return R3;
				}

				/**
				 * @brief Compute R̃_{ij}^{(4)} = γ̃^{ℓ m} [ 2 Γ̃^k_{ℓ(i} Γ̃_{j) k m} + Γ̃^k_{i m} Γ̃_{k ℓ j} ]
				 *
				 * @param gamma_tilde_inv    Inverse conformal metric γ̃^{ℓ m}
				 * @param christoffel_tilde  Conformal Christoffel symbols Γ̃^k_{ i j }
				 * @return tensorium::Tensor<T,2>  The R^{(4)}_{ij} contribution
				 */
				static tensorium::Tensor<T,2> compute_GammaProduct_term(
						const tensorium::Tensor<T,2>& gamma_tilde_inv,
						const tensorium::Tensor<T,3>& christoffel_tilde)
				{
					using namespace tensorium;
					Tensor<T,2> R4({3,3});

					for (size_t i = 0; i < 3; ++i) {
						for (size_t j = 0; j < 3; ++j) {
							T sum = 0;
							// ℓ,m index over 0..2
							for (size_t ell = 0; ell < 3; ++ell) {
								for (size_t m = 0; m < 3; ++m) {
									T term1 = 0;
									for (size_t k = 0; k < 3; ++k) {
										term1 += christoffel_tilde(k, ell, i)
											* christoffel_tilde(j  , k  , m);
									}
									for (size_t k = 0; k < 3; ++k) {
										term1 += christoffel_tilde(k, ell, j)
											* christoffel_tilde(i  , k  , m);
									}
									T term2 = 0;
									for (size_t k = 0; k < 3; ++k) {
										term2 += christoffel_tilde(k, i  , m)
											* christoffel_tilde(j, k  , ell);
									}
									sum += gamma_tilde_inv(ell, m) * (term1 + term2);
								}
							}
							R4(i, j) = sum;
						}
					}
					return R4;
				}
		};


}
