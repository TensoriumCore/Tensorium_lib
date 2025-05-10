#pragma once

#include "../Core/Vector.hpp"
#include "../Core/Tensor.hpp"
#include "../Core/Matrix.hpp"
#include "../Functionnal/FunctionnalRG.hpp"
#include "../DiffGeometry/ChristoffelSymbol.hpp"
#include "../DiffGeometry/Metric.hpp"
#include "../Core/Derivate.hpp"


namespace morpheus_RG {
/**
 * @brief Computes the 4D Riemann curvature tensor \f$ R^\rho_{\ \sigma\mu\nu} \f$
 *
 * The following expression is used:
 * \f[
 * R^\rho_{\ \sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma}
 * - \partial_\nu \Gamma^\rho_{\mu\sigma}
 * + \Gamma^\rho_{\mu\lambda} \Gamma^\lambda_{\nu\sigma}
 * - \Gamma^\rho_{\nu\lambda} \Gamma^\lambda_{\mu\sigma}
 * \f]
 */

	template <typename T>
		class RiemannTensor {
			public:
				using Tensor4D = morpheus::Tensor<T, 4>;
				using Tensor2D = morpheus::Tensor<T, 2>;
				using VectorT  = morpheus::Vector<T>;


				static void print_componentwise(const Tensor4D& R, T threshold = 1e-12) {
					constexpr size_t dim = 4;
					std::cout << "\nRiemann tensor components (sliced by upper index λ):\n";

					for (size_t lambda = 0; lambda < dim; ++lambda) {
						std::cout << "R^" << lambda << "_{μνρ} :\n";
						for (size_t mu = 0; mu < dim; ++mu) {
							for (size_t nu = 0; nu < dim; ++nu) {
								std::cout << "    ";
								for (size_t rho = 0; rho < dim; ++rho) {
									T val = R({lambda, mu, nu, rho});
									if (std::abs(val) < threshold)
										val = 0.0;
									std::cout << std::setw(10) << std::fixed << std::setprecision(6) << val << " ";
								}
								std::cout << "\n";
							}
						}
						std::cout << "\n";
					}
				}

				static void print(const Tensor4D& R, const std::string& name = "Riemann") {
					constexpr size_t dim = 4;
					std::cout << name << " tensor components:\n";
					for (size_t rho = 0; rho < dim; ++rho)
						for (size_t sigma = 0; sigma < dim; ++sigma)
							for (size_t mu = 0; mu < dim; ++mu)
								for (size_t nu = 0; nu < dim; ++nu) {
									const T val = R({rho, sigma, mu, nu});
									if (std::abs(val) > T(1e-12)) {
										std::cout << name << "[" << rho << "][" << sigma << "]["
											<< mu << "][" << nu << "] = " << val << "\n";
									}
								}
				}
				/**
				 * @brief Computes the 4D Riemann curvature tensor \f$ R^\rho_{\ \sigma\mu\nu} \f$
				 *
				 * This function numerically computes the full 4D Riemann tensor using finite difference approximations
				 * (via Richardson extrapolation) on the Christoffel symbols computed from the metric tensor.
				 *
				 * The Riemann tensor is defined as:
				 * \f[
				 * R^\rho_{\ \sigma\mu\nu} =
				 * \partial_\mu \Gamma^\rho_{\nu\sigma}
				 * - \partial_\nu \Gamma^\rho_{\mu\sigma}
				 * + \Gamma^\rho_{\mu\lambda} \Gamma^\lambda_{\nu\sigma}
				 * - \Gamma^\rho_{\nu\lambda} \Gamma^\lambda_{\mu\sigma}
				 * \f]
				 *
				 * @tparam T Scalar type (e.g., float or double)
				 * @param X Coordinate vector \f$ X^\mu \f$ (4D)
				 * @param h Spacing for finite difference approximations
				 * @param metric A callable metric object (e.g. Kerr, Schwarzschild) implementing operator()(X, g)
				 * @return A 4th-rank tensor \f$ R^\rho_{\ \sigma\mu\nu} \f$ stored as a Tensor4D object
				 */
				static Tensor4D compute(const VectorT& X, T h, const Metric<T>& metric) {
					constexpr size_t dim = 4;
					Tensor2D g({dim, dim});
					metric(X, g);
					Tensor2D g_inv = morpheus_RG::inv_mat_tensor_local(g);

					std::array<ChristoffelSym<T>, 4> gamma_ph {
						ChristoffelSym<T>(h), ChristoffelSym<T>(h),
							ChristoffelSym<T>(h), ChristoffelSym<T>(h)
					};
					std::array<ChristoffelSym<T>, 4> gamma_mh = gamma_ph;
					std::array<ChristoffelSym<T>, 4> gamma_ph2 = gamma_ph;
					std::array<ChristoffelSym<T>, 4> gamma_mh2 = gamma_ph;

					for (size_t mu = 0; mu < dim; ++mu) {
						VectorT Xh = X, Xl = X, Xh2 = X, Xl2 = X;
						Xh(mu)  += h;
						Xl(mu)  -= h;
						Xh2(mu) += h * 0.5;
						Xl2(mu) -= h * 0.5;

						Tensor2D gh({dim, dim}), gl({dim, dim});
						Tensor2D gh2({dim, dim}), gl2({dim, dim});

						metric(Xh, gh);     metric(Xl, gl);
						metric(Xh2, gh2);   metric(Xl2, gl2);

						auto inv_gh  = morpheus_RG::inv_mat_tensor_local(gh);
						auto inv_gl  = morpheus_RG::inv_mat_tensor_local(gl);
						auto inv_gh2 = morpheus_RG::inv_mat_tensor_local(gh2);
						auto inv_gl2 = morpheus_RG::inv_mat_tensor_local(gl2);

						gamma_ph[mu]  = compute_christoffel(Xh,  h, gh,  inv_gh,  metric);
						gamma_mh[mu]  = compute_christoffel(Xl,  h, gl,  inv_gl,  metric);
						gamma_ph2[mu] = compute_christoffel(Xh2, h, gh2, inv_gh2, metric);
						gamma_mh2[mu] = compute_christoffel(Xl2, h, gl2, inv_gl2, metric);
					}

					Tensor4D R({dim, dim, dim, dim});
					R.fill(T(0));

					for (size_t rho = 0; rho < dim; ++rho)
						for (size_t sigma = 0; sigma < dim; ++sigma)
							for (size_t mu = 0; mu < dim; ++mu)
								for (size_t nu = 0; nu < dim; ++nu) {
									T dGamma_mu = morpheus::richardson_derivative(
											gamma_ph[mu](rho, nu, sigma, 0),
											gamma_mh[mu](rho, nu, sigma, 0),
											gamma_ph2[mu](rho, nu, sigma, 0),
											gamma_mh2[mu](rho, nu, sigma, 0),
											h);

									T dGamma_nu = morpheus::richardson_derivative(
											gamma_ph[nu](rho, mu, sigma, 0),
											gamma_mh[nu](rho, mu, sigma, 0),
											gamma_ph2[nu](rho, mu, sigma, 0),
											gamma_mh2[nu](rho, mu, sigma, 0),
											h);

									T sum = dGamma_mu - dGamma_nu;

									for (size_t lambda = 0; lambda < dim; ++lambda) {
										sum += gamma_ph[mu](rho, mu, lambda, 0) * gamma_ph[nu](lambda, nu, sigma, 0);
										sum -= gamma_ph[nu](rho, nu, lambda, 0) * gamma_ph[mu](lambda, mu, sigma, 0);
									}

									R({rho, sigma, mu, nu}) = sum;
								}

					return R;
				}
		};

}
