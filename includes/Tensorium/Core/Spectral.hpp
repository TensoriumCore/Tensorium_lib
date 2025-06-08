#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cassert>
#include <array>
#include <stdexcept>
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"
#include "Vector.hpp"
#include "Tensor.hpp"
#include "Matrix.hpp"
#include "../MathUtils/MathsUtils.hpp"
#include <numbers>

namespace tensorium {
	/**
	 * @brief Fast Fourier Transform (FFT) implementation using Cooley–Tukey algorithm
	 *
	 * Performs in-place FFT or inverse FFT on complex-valued vectors whose size is a power of 2.
	 *
	 * @tparam T Underlying scalar type (e.g., float or double)
	 */
	template<typename T>
		class SpectralFFT {
			public:
				using Tensor2D  = tensorium::Tensor<T, 2>;
				using VectorT   = tensorium::Vector<T>;
				using C         = std::complex<T>;
				using CVectorT  = tensorium::Vector<C>;
				/**
				 * @brief Perform forward FFT (in-place)
				 *
				 * @param a Input/output complex vector (must have power-of-two size)
				 */
				static inline void forward(CVectorT& a)  { transform_impl(a, false); }

				static void forward_3D(Tensor<std::complex<T>, 3>& a) {
					const auto shape = a.shape();
					const size_t NX = shape[0], NY = shape[1], NZ = shape[2];
					using CVector = Vector<std::complex<T>>;

#pragma omp parallel for collapse(2)
					for (size_t i = 0; i < NX; ++i) {
						for (size_t j = 0; j < NY; ++j) {
							CVector sliceZ(NZ);
							for (size_t k = 0; k < NZ; ++k)
								sliceZ(k) = a(i,j,k);

							forward(sliceZ);

							for (size_t k = 0; k < NZ; ++k)
								a(i,j,k) = sliceZ(k);
						}
					}

#pragma omp parallel for collapse(2)
					for (size_t i = 0; i < NX; ++i) {
						for (size_t k = 0; k < NZ; ++k) {
							CVector sliceY(NY);
							for (size_t j = 0; j < NY; ++j)
								sliceY(j) = a(i,j,k);

							forward(sliceY);

							for (size_t j = 0; j < NY; ++j)
								a(i,j,k) = sliceY(j);
						}
					}

#pragma omp parallel for collapse(2)
					for (size_t j = 0; j < NY; ++j) {
						for (size_t k = 0; k < NZ; ++k) {
							CVector sliceX(NX);
							for (size_t i = 0; i < NX; ++i)
								sliceX(i) = a(i,j,k);

							forward(sliceX);

							for (size_t i = 0; i < NX; ++i)
								a(i,j,k) = sliceX(i);
						}
					}
				}

				/**
				 * @brief Perform inverse FFT (in-place)
				 *
				 * @param a Input/output complex vector (must have power-of-two size)
				 */
				static inline void backward(CVectorT& a) { transform_impl(a, true ); }

				static void backward_3D(Tensor<std::complex<T>, 3>& a) {
					const auto shape = a.shape();
					const size_t NX = shape[0], NY = shape[1], NZ = shape[2];
					using CVector = Vector<std::complex<T>>;

#pragma omp parallel for collapse(2)
					for (size_t j = 0; j < NY; ++j) {
						for (size_t k = 0; k < NZ; ++k) {
							CVector sliceX(NX);
							for (size_t i = 0; i < NX; ++i)
								sliceX(i) = a(i,j,k);

							backward(sliceX);

							for (size_t i = 0; i < NX; ++i)
								a(i,j,k) = sliceX(i);
						}
					}

#pragma omp parallel for collapse(2)
					for (size_t i = 0; i < NX; ++i) {
						for (size_t k = 0; k < NZ; ++k) {
							CVector sliceY(NY);
							for (size_t j = 0; j < NY; ++j)
								sliceY(j) = a(i,j,k);

							backward(sliceY);

							for (size_t j = 0; j < NY; ++j)
								a(i,j,k) = sliceY(j);
						}
					}

#pragma omp parallel for collapse(2)
					for (size_t i = 0; i < NX; ++i) {
						for (size_t j = 0; j < NY; ++j) {
							CVector sliceZ(NZ);
							for (size_t k = 0; k < NZ; ++k)
								sliceZ(k) = a(i,j,k);

							backward(sliceZ);

							for (size_t k = 0; k < NZ; ++k)
								a(i,j,k) = sliceZ(k);
						}
					}
				}

			private:
				/**
				 * @brief Internal FFT implementation (shared by forward/backward)
				 * @param a Vector to transform
				 * @param inverse Whether to perform inverse FFT
				 */
				static void transform_impl(CVectorT& a, bool inverse)
				{
					const std::size_t N = a.size();
					if (N <= 1 || (N & (N - 1)))
						throw std::invalid_argument("SpectralFFT: size must be power of two");

					bit_reverse(a);

					std::vector<C> twiddles(N / 2);
					const T sign = inverse ? T(+1) : T(-1);
					constexpr T pi = T(3.141592653589793238462643383279502884L);
					for (std::size_t k = 0; k < N / 2; ++k)
						twiddles[k] = { std::cos(sign * 2 * pi * k / N), std::sin(sign * 2 * pi * k / N) };

					for (std::size_t len = 2; len <= N; len <<= 1) {
						const std::size_t half = len >> 1;
						const std::size_t step = N / len;

#pragma omp parallel for schedule(static) if(len >= 64)
						for (std::size_t i = 0; i < N; i += len) {
							for (std::size_t j = 0; j < half; ++j) {
								C t = a[i + j + half] * twiddles[j * step];
								C u = a[i + j];
								a[i + j]        = u + t;
								a[i + j + half] = u - t;
							}
						}
					}

					if (inverse) {
						const T invN = T(1) / T(N);
#pragma omp parallel for schedule(static)
						for (std::size_t i = 0; i < N; ++i) a[i] *= invN;
					}
				}
				/**
				 * @brief Bit-reversal permutation step
				 * @param a Vector to reorder
				 */
				static void bit_reverse(CVectorT& a)
				{
					const std::size_t N = a.size();
					for (std::size_t i = 1, j = 0; i < N; ++i) {
						std::size_t bit = N >> 1;
						for (; j & bit; bit >>= 1) j ^= bit;
						j ^= bit;
						if (i < j) std::swap(a[i], a[j]);
					}
				}
		};
	/**
	 * @brief Placeholder Chebyshev spectral method class
	 *
	 * This is a demonstration implementation for Chebyshev-based operator application.
	 *
	 * @tparam T Scalar type
	 */
	template<typename T>
		class SpectalChebyshev {
			public:
				using Tensor2D = tensorium::Tensor<T, 2>;
				using VectorT  = tensorium::Vector<T>;

				/**
				 * @brief Dummy computation using Chebyshev-like cosine weights
				 *
				 * Fills the result matrix with \f$ \cos(X_i X_j) \cdot h \f$
				 *
				 * @param X Input vector (length must match dim)
				 * @param h Scaling factor
				 * @param result Output 2D tensor
				 */
				static void compute(const VectorT& X, T h, Tensor2D& result) {
					const size_t dim = 4;
					result.resize(dim, dim);
					result.fill(T(0));

					for (size_t i = 0; i < dim; ++i) {
						for (size_t j = 0; j < dim; ++j) {
							result(i, j) = std::cos(X(i) * X(j)) * h;
						}
					}
				}
		};			
}

