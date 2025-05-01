#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include <array>
#include <stdexcept>
#include <iomanip>
#include <functional>
#include "../SIMD/SIMD.hpp"
#include "../SIMD/Allocator.hpp"
#include "../Core/Matrix.hpp"
#include "../Core/Vector.hpp"
#include "../Core/Tensor.hpp"

namespace morpheus_RG {

	template<typename T>
		class ChristoffelSym {
			public:
				aligned_vector<T> data;
				static constexpr size_t rank = 4;
				size_t dim;

				ChristoffelSym(size_t dim) : dim(dim), data(dim * dim * dim * dim) {}

				T& operator()(size_t i, size_t j, size_t k, size_t l) {
					return data[i * dim * dim * dim + j * dim * dim + k * dim + l];
				}

				const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
					return data[i * dim * dim * dim + j * dim * dim + k * dim + l];
				}

				void fill(T value) {
					std::fill(data.begin(), data.end(), value);
				}


				void print() const {
					for (size_t l = 0; l < dim; ++l) {
						std::cout << "Γ^" << l << "_{μν} :\n";
						for (size_t i = 0; i < dim; ++i) {
							for (size_t j = 0; j < dim; ++j) {
								std::cout << std::setw(12) << std::setprecision(6) << std::fixed << (*this)(l, i, j, 0) << " ";
							}
							std::cout << "\n";
						}
						std::cout << "\n";
					}
				}


				__attribute__((always_inline, hot, flatten))
					static inline ChristoffelSym<T> compute_christoffel(
							const morpheus::Vector<T>& X, T h,
							const morpheus::Tensor<T, 2>& g,
							const morpheus::Tensor<T, 2>& g_inv,
							const std::function<void(const morpheus::Vector<T>&, morpheus::Tensor<T, 2>&)>& metric_generator)
					{
						const size_t dim = X.size();
						ChristoffelSym<T> gamma(dim);
						ChristoffelSym<T> d_metric(dim);
						gamma.fill(T(0));
						d_metric.fill(T(0));

						morpheus::Vector<T> Xh = X, Xl = X;
						morpheus::Tensor<T, 2> gh({dim, dim}), gl({dim, dim});

						for (size_t mu = 0; mu < dim; ++mu) {
							Xh = X;
							Xl = X;
							Xh(mu) += h;
							Xl(mu) -= h;

							metric_generator(Xh, gh);
							metric_generator(Xl, gl);

							for (size_t nu = 0; nu < dim; ++nu)
								for (size_t lam = 0; lam < dim; ++lam)
									d_metric(lam, nu, mu, 0) = (gh(lam, nu) - gl(lam, nu)) / (T(2) * h);
						}

						ChristoffelSym<T> tmp(dim);
						tmp.fill(T(0));
						for (size_t lam = 0; lam < dim; ++lam)
							for (size_t nu = 0; nu < dim; ++nu)
								for (size_t mu = 0; mu < dim; ++mu)
									tmp(lam, nu, mu, 0) = T(0.5) * (
											d_metric(nu, lam, mu, 0) +
											d_metric(mu, lam, nu, 0) -
											d_metric(mu, nu, lam, 0));

						for (size_t lam = 0; lam < dim; ++lam)
							for (size_t nu = 0; nu < dim; ++nu)
								for (size_t mu = 0; mu < dim; ++mu) {
									T sum = T(0);
									for (size_t kap = 0; kap < dim; ++kap)
										sum += g_inv(lam, kap) * tmp(kap, nu, mu, 0);
									gamma(lam, nu, mu, 0) = sum;
								}

						return gamma;
					}
		};

} 
