#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace tensorium_RG {
	/**
	 * @class BSSNChristoffel
	 * @brief Compute the conformal Christoffel symbols \f$ \tilde{\Gamma}^k_{ij} \f$
	 *
	 * The conformal Christoffel symbols are computed from the inverse conformal metric 
	 * \f$ \tilde{\gamma}^{kl} \f$ and the derivatives of the conformal metric \f$ \partial_m \tilde{\gamma}_{ij} \f$
	 * using the formula:
	 *
	 * \f[
	 * \tilde{\Gamma}^k_{ij} =
	 * \frac{1}{2} \tilde{\gamma}^{kl}
	 * \left( \partial_i \tilde{\gamma}_{jl}
	 *      + \partial_j \tilde{\gamma}_{il}
	 *      - \partial_l \tilde{\gamma}_{ij} \right)
	 * \f]
	 */
	template<typename T>
		class BSSNChristoffel {
			public:
				/**
				 * @brief Compute the conformal Christoffel symbols \f$ \tilde{\Gamma}^k_{ij} \f$
				 *
				 * @param gamma_tilde       3×3 conformal metric \f$ \tilde{\gamma}_{ij} \f$
				 * @param dgamma_tilde      3×3×3 partial derivatives \f$ \partial_k \tilde{\gamma}_{ij} \f$
				 * @param gamma_tilde_inv   3×3 inverse metric \f$ \tilde{\gamma}^{ij} \f$
				 * @param Christoffel       Output: \f$ \tilde{\Gamma}^k_{ij} \f$
				 */
				static void compute(
						const tensorium::Tensor<T, 2>& gamma_tilde,
						const tensorium::Tensor<T, 3>& dgamma_tilde,
						const tensorium::Tensor<T, 2>& gamma_tilde_inv,
						tensorium::Tensor<T, 3>& Christoffel
						) {

					if (gamma_tilde.shape() != std::array<size_t,2>{3,3})
						throw std::invalid_argument("gamma_tilde must be 3x3");
					if (gamma_tilde_inv.shape() != std::array<size_t,2>{3,3})
						throw std::invalid_argument("gamma_tilde_inv must be 3x3");
					if (dgamma_tilde.shape() != std::array<size_t,3>{3,3,3})
						throw std::invalid_argument("dgamma_tilde must be 3x3x3");


					Christoffel.resize(3, 3, 3);

					for (size_t k = 0; k < 3; ++k) {
						for (size_t i = 0; i < 3; ++i) {
							for (size_t j = 0; j < 3; ++j) {
								T sum = 0.0;
								for (size_t l = 0; l < 3; ++l) {
									T term = dgamma_tilde(i, l, j) + dgamma_tilde(j, l, i) - dgamma_tilde(l, i, j);
									sum += gamma_tilde_inv(k, l) * term;
								}
								Christoffel(k, i, j) = T(0.5) * sum;
							}
						}
					}
				}

				static void compute3D(
						const tensorium::Tensor<T, 5>& gamma_tilde,        // [NX, NY, NZ, 3, 3]
						const tensorium::Tensor<T, 6>& dgamma_tilde,       // [NX, NY, NZ, 3, 3, 3]
						const tensorium::Tensor<T, 5>& gamma_tilde_inv,    // [NX, NY, NZ, 3, 3]
						tensorium::Tensor<T, 6>& Christoffel               // [NX, NY, NZ, 3, 3, 3] (output)
						) {
					using namespace tensorium;

					const size_t NX = gamma_tilde.dim(0);
					const size_t NY = gamma_tilde.dim(1);
					const size_t NZ = gamma_tilde.dim(2);

					Christoffel.resize(NX, NY, NZ, 3, 3, 3);

					for (size_t x = 0; x < NX; ++x) {
						for (size_t y = 0; y < NY; ++y) {
							for (size_t z = 0; z < NZ; ++z) {
								Tensor<T, 2> gtilde(3, 3);
								Tensor<T, 2> ginv(3, 3);
								Tensor<T, 3> dg(3, 3, 3);
								Tensor<T, 3> chr(3, 3, 3);

								for (size_t i = 0; i < 3; ++i)
									for (size_t j = 0; j < 3; ++j) {
										gtilde(i, j) = gamma_tilde(x, y, z, i, j);
										ginv(i, j) = gamma_tilde_inv(x, y, z, i, j);
										for (size_t k = 0; k < 3; ++k)
											dg(i, j, k) = dgamma_tilde(x, y, z, i, j, k);
									}

								BSSNChristoffel<T>::compute(gtilde, dg, ginv, chr);

								for (size_t i = 0; i < 3; ++i)
									for (size_t j = 0; j < 3; ++j)
										for (size_t k = 0; k < 3; ++k)
											Christoffel(x, y, z, i, j, k) = chr(i, j, k);
							}
						}
					}
				}

		};
	/**
	 * @brief Compute the 3D partial derivatives \f$ \partial_k \tilde{\gamma}_{ij} \f$ from a 5D field tensor
	 *
	 * Uses centered 4th-order or 2nd-order finite differences for inner and boundary points.
	 *
	 * \f[
	 * \partial_k \tilde{\gamma}_{ij} =
	 * \begin{cases}
	 * \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} & \text{if inner} \\
	 * \frac{f(x+h) - f(x-h)}{2h} & \text{if near boundary} \\
	 * 0 & \text{otherwise}
	 * \end{cases}
	 * \f]
	 *
	 * @param gamma_field      Field of type Tensor<T, 5> with dimensions [Nx, Ny, Nz, 3, 3]
	 * @param i,j,k            Grid position
	 * @param dx,dy,dz         Grid spacings
	 * @param dgamma_out       Output tensor \f$ \partial_k \tilde{\gamma}_{ij} \f$
	 */
	template<typename T>
		inline void compute_partial_derivatives_3D(
				const tensorium::Tensor<T, 5>& gamma_field,
				size_t i, size_t j, size_t k,
				T dx, T dy, T dz,
				tensorium::Tensor<T, 3>& dgamma_out) {

			dgamma_out.resize(3, 3, 3);

			auto get = [&](int ii, int jj, int kk, int a, int b) -> T {
				return gamma_field(ii, jj, kk, a, b);
			};

			auto dim = gamma_field.shape();
			size_t Nx = dim[0], Ny = dim[1], Nz = dim[2];

			for (size_t a = 0; a < 3; ++a) {
				for (size_t b = 0; b < 3; ++b) {
					if (i >= 2 && i + 2 < Nx)
						dgamma_out(0, a, b) = (-get(i+2,j,k,a,b) + 8*get(i+1,j,k,a,b) - 8*get(i-1,j,k,a,b) + get(i-2,j,k,a,b)) / (12 * dx);
					else if (i > 0 && i + 1 < Nx)
						dgamma_out(0, a, b) = (get(i+1,j,k,a,b) - get(i-1,j,k,a,b)) / (2 * dx);
					else
						dgamma_out(0, a, b) = T(0);

						if (j >= 2 && j + 2 < Ny)
							dgamma_out(1, a, b) = (-get(i,j+2,k,a,b) + 8*get(i,j+1,k,a,b) - 8*get(i,j-1,k,a,b) + get(i,j-2,k,a,b)) / (12 * dy);
						else if (j > 0 && j + 1 < Ny)
							dgamma_out(1, a, b) = (get(i,j+1,k,a,b) - get(i,j-1,k,a,b)) / (2 * dy);
						else
							dgamma_out(1, a, b) = T(0);

						if (k >= 2 && k + 2 < Nz)
							dgamma_out(2, a, b) = (-get(i,j,k+2,a,b) + 8*get(i,j,k+1,a,b) - 8*get(i,j,k-1,a,b) + get(i,j,k-2,a,b)) / (12 * dz);
						else if (k > 0 && k + 1 < Nz)
							dgamma_out(2, a, b) = (get(i,j,k+1,a,b) - get(i,j,k-1,a,b)) / (2 * dz);
						else
							dgamma_out(2, a, b) = T(0);
					}
				}
			}
} // namespace tensorium::bssn
