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

namespace morpheus {
	template<typename T>
		class SpectralFFT {
			public:
				using Tensor2D  = morpheus::Tensor<T, 2>;
				using VectorT   = morpheus::Vector<T>;
				using C         = std::complex<T>;
				using CVectorT  = morpheus::Vector<C>;

				static inline void forward(CVectorT& a)  { transform_impl(a, false); }
				static inline void backward(CVectorT& a) { transform_impl(a, true ); }

			private:
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

	template<typename T>
		class SpectalChebyshev {
			public:
				using Tensor2D = morpheus::Tensor<T, 2>;
				using VectorT  = morpheus::Vector<T>;

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

