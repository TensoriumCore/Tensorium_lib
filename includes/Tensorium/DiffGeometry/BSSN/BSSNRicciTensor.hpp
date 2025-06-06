#pragma once

#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Vector.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNContractedChristoffel.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {

	template<typename T>
		class RicciTensor3D {
			public:

				static tensorium::Tensor<T, 5> compute_ricci_conformal(
						const tensorium::Tensor<T, 5>& gamma_tilde,
						const tensorium::Tensor<T, 5>& gamma_tilde_inv,
						const tensorium::Tensor<T, 6>& christoffel_tilde,
						const tensorium::Tensor<T, 5>& Gamma_deriv, 
						const tensorium::Tensor<T, 7>& d2gamma_tilde) 
				{
					const auto& shape = gamma_tilde.shape();
					const size_t NX = shape[0], NY = shape[1], NZ = shape[2];
					tensorium::Tensor<T, 5> Ricci_tilde({NX, NY, NZ, 3, 3});

#pragma omp parallel for
					for (size_t x = 1; x < NX-1; ++x) {
						for (size_t y = 1; y < NY-1; ++y) {
							for (size_t z = 1; z < NZ-1; ++z) {

								const auto gamma = [&](size_t i, size_t j) { 
									return gamma_tilde({x, y, z, i, j}); 
								};
								const auto dGamma = [&](size_t i) { 
									return Gamma_deriv({x, y, z, i}); 
								};

								for (size_t i = 0; i < 3; ++i) {
									for (size_t j = 0; j < 3; ++j) {
										T term_laplacian = 0;
										for (size_t k = 0; k < 3; ++k) {
											for (size_t l = 0; l < 3; ++l) {
												term_laplacian += gamma_tilde_inv({x,y,z,k,l}) 
													* d2gamma_tilde({x,y,z,i,j,k,l});
											}
										}
										term_laplacian *= -0.5;

										T term_divGamma = gamma(i,0)*Gamma_deriv({x,y,z,j,0}) 
											+ gamma(j,0)*Gamma_deriv({x,y,z,i,0})
											+ gamma(i,1)*Gamma_deriv({x,y,z,j,1})
											+ gamma(j,1)*Gamma_deriv({x,y,z,i,1})
											+ gamma(i,2)*Gamma_deriv({x,y,z,j,2})
											+ gamma(j,2)*Gamma_deriv({x,y,z,i,2});

										T term_gamma_gamma = 0;
										for (size_t m = 0; m < 3; ++m) {
											for (size_t k = 0; k < 3; ++k) {
												term_gamma_gamma -= christoffel_tilde({x,y,z,m,k,i}) 
													* christoffel_tilde({x,y,z,k,m,j});
											}
										}

										Ricci_tilde({x,y,z,i,j}) = term_laplacian 
											+ 0.5 * term_divGamma 
											+ term_gamma_gamma;
									}
								}

								for (size_t i = 0; i < 3; ++i) {
									for (size_t j = i+1; j < 3; ++j) {
										T avg = 0.5 * (Ricci_tilde({x,y,z,i,j}) 
												+ Ricci_tilde({x,y,z,j,i}));
										Ricci_tilde({x,y,z,i,j}) = Ricci_tilde({x,y,z,j,i}) = avg;
									}
								}
							}
						}
					}
					return Ricci_tilde;
				}
		};

	template <typename T>
		tensorium::Tensor<T, 2> compute_ricci_tilde_at(
				const tensorium::Vector<T>& X,
				const tensorium_RG::Metric<T>& metric,
				T dx, T dy, T dz)
		{
			constexpr size_t NX = 3, NY = 3, NZ = 3;

			tensorium::Tensor<T, 5> gamma_tilde({NX, NY, NZ, 3, 3});
			tensorium::Tensor<T, 5> gamma_tilde_inv({NX, NY, NZ, 3, 3});
			tensorium::Tensor<T, 6> dgamma_tilde({NX, NY, NZ, 3, 3, 3});
			tensorium::Tensor<T, 7> d2gamma_tilde({NX, NY, NZ, 3, 3, 3, 3});

			T a; tensorium::Vector<T> b(3); tensorium::Tensor<T,2> g({3,3});
									T chi = compute_conformal_factor(metric, g);

			tensorium::Tensor<T,2> gtilde = compute_conformal_metric(metric, g, chi);

			for (size_t i = 0; i < NX; ++i)
				for (size_t j = 0; j < NY; ++j)
					for (size_t k = 0; k < NZ; ++k) {
						tensorium::Vector<T> Xs = {
							X(0) + dx * (i - NX/2),
							X(1) + dy * (j - NY/2),
							X(2) + dz * (k - NZ/2)
						};
						metric.BSSN(Xs, a, b, g);
						T chi = compute_conformal_factor(metric, g);
						tensorium::Tensor<T,2> gtilde = compute_conformal_metric(metric, g, chi);
						gamma_tilde(i,j,k) = gtilde;
						gamma_tilde_inv(i,j,k) = inv_mat_tensor(gtilde);
					}

			tensorium_RG::compute_partial_derivatives_tensor2D<T>(
					X, dx, dy, dz,
					[&](const tensorium::Vector<T>& Xs, tensorium::Tensor<T,2>& out) {
					T a; tensorium::Vector<T> b(3); tensorium::Tensor<T,2> g({3,3});
					metric.BSSN(Xs, a, b, g);
					T chi = compute_conformal_factor(metric, g);
					out = compute_conformal_metric(metric, g, chi);
					}, dgamma_tilde
					);

			tensorium_RG::compute_second_derivatives_tensor2D<T>(
					X, dx, dy, dz,
					[&](const tensorium::Vector<T>& Xs, tensorium::Tensor<T,2>& out) {
					T a; 
					tensorium::Vector<T> b(3); tensorium::Tensor<T,2> g({3,3});
					metric.BSSN(Xs, a, b, g);
					T chi = compute_conformal_factor(metric, g);
					out = compute_conformal_metric(metric, g, chi);
					}, d2gamma_tilde
					);

			tensorium::Tensor<T, 6> christoffel_tilde({NX, NY, NZ, 3, 3, 3});
			for (size_t i = 0; i < NX; ++i)
				for (size_t j = 0; j < NY; ++j)
					for (size_t k = 0; k < NZ; ++k)
						tensorium_RG::BSSNChristoffel<T>::compute(gamma_tilde, dgamma_tilde, gamma_tilde_inv, christoffel_tilde);


			tensorium::Tensor<T, 5> Gamma_deriv({NX, NY, NZ, 3, 3});
			 tensorium_RG::BSSNContractedGamma<T>::compute(
							X, metric, dx, dy, dz, chi);

			auto Ricci_full = tensorium_RG::RicciTensor3D<T>::compute_ricci_conformal(
					gamma_tilde,
					gamma_tilde_inv,
					christoffel_tilde,
					Gamma_deriv,
					d2gamma_tilde
					);

			tensorium::Tensor<T, 2> R({3, 3});
			for (size_t i = 0; i < 3; ++i)
				for (size_t j = 0; j < 3; ++j)
					R(i,j) = Ricci_full(1,1,1,i,j);

			return R;
		}
}
