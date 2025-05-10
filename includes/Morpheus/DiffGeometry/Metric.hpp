#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "../Core/Tensor.hpp"
#include "../Core/Vector.hpp"
#include <functional>
namespace morpheus_RG {

	template<typename T>
		class Metric {
			public:
				std::string type;
				T M = T(1.0); 
				T a = T(0.935);

				Metric(const std::string& metric_type = "minkowski", T mass = T(1.0), T spin = T(0.0))
					: type(metric_type), M(mass), a(spin) {}

				void operator()(const morpheus::Vector<T>& X, morpheus::Tensor<T, 2>& g) const {
					if (type == "minkowski") {
						compute_minkowski(g);
					} else if (type == "schwarzschild") {
						compute_schwarzschild(X, g);
					} else if (type == "kerr") {
						compute_kerr(X, g);
					} else if (type == "flrw") {
						compute_flrw(X, g);
					} else if (type == "kerr_schild") {
							compute_kerr_schild(X, g);
					} else if (type == "custom" && custom_metric_fn) {
						custom_metric_fn(X, g);
					} else {
						throw std::invalid_argument("Unknown metric type: " + type);
					}
				}

				inline void BSSN(const morpheus::Vector<T>& X,
						T& alpha,
						morpheus::Vector<T>& beta,
						morpheus::Tensor<T, 2>& gamma) const
				{
					assert(X.size() == 4);
					morpheus::Tensor<T, 2> g({4, 4});
					(*this)(X, g);

					alpha = std::sqrt(-g({0, 0}));

					beta.resize(3);
					for (size_t i = 0; i < 3; ++i)
						beta(i) = g(0, i + 1); 

					gamma.resize(3, 3);
					for (size_t i = 0; i < 3; ++i)
						for (size_t j = 0; j < 3; ++j)
							gamma(i, j) = g(i + 1, j + 1);
				}
			private:
				std::function<void(const morpheus::Vector<T>&, morpheus::Tensor<T, 2>&)> custom_metric_fn = nullptr;

				void set_custom(std::function<void(const morpheus::Vector<T>&, morpheus::Tensor<T, 2>&)> fn) {
					custom_metric_fn = std::move(fn);
					type = "custom";
				}

				void compute_minkowski(morpheus::Tensor<T, 2>& g) const {
					const size_t dim = 4;
					g.resize(dim, dim);
					g.fill(T(0));
					g(0, 0) = T(-1);
					g(1, 1) = T(1);
					g(2, 2) = T(1);
					g(3, 3) = T(1);
				}

				void compute_schwarzschild(const morpheus::Vector<T>& X, morpheus::Tensor<T, 2>& g) const {
					assert(X.size() == 4);
					const T r = X(1);
					const T theta = X(2);
					const T sin_theta = std::sin(theta);
					const T f = T(1) - T(2) * M / r;

					g.resize(4, 4);
					g.fill(T(0));
					g(0, 0) = -f;
					g(1, 1) = T(1) / f;
					g(2, 2) = r * r;
					g(3, 3) = r * r * sin_theta * sin_theta;
				}

				void compute_kerr(const morpheus::Vector<T>& X, morpheus::Tensor<T, 2>& g) const {
					assert(X.size() == 4);
					const T r = X(1);
					const T theta = X(2);
					const T sin_theta = std::sin(theta);
					const T cos_theta = std::cos(theta);
					const T sin2 = sin_theta * sin_theta;

					const T Sigma = r * r + a * a * cos_theta * cos_theta;
					const T Delta = r * r - T(2) * M * r + a * a;

					g.resize({4, 4});
					g.fill(T(0));

					g(0, 0) = -(T(1) - (T(2) * M * r) / Sigma);
					g(0, 3) = g(3, 0) = -T(2) * M * r * a * sin2 / Sigma;
					g(1, 1) = Sigma / Delta;
					g(2, 2) = Sigma;
					g(3, 3) = (r * r + a * a + T(2) * M * r * a * a * sin2 / Sigma) * sin2;
				}

				void compute_flrw(const morpheus::Vector<T>& X, morpheus::Tensor<T, 2>& g) const {
					assert(X.size() == 4);
					const T t = X(0);
					const T r = X(1);
					const T theta = X(2);
					const T sin_theta = std::sin(theta);

					const T a_t = std::pow(t, T(2.0) / T(3.0));

					g.resize(4, 4);
					g.fill(T(0));

					g(0, 0) = T(-1);
					g(1, 1) = a_t * a_t;
					g(2, 2) = a_t * a_t * r * r;
					g(3, 3) = a_t * a_t * r * r * sin_theta * sin_theta;
				}

				T kerr_schild_radius(T x, T y, T z) const {
					const T r2 = x * x + y * y + z * z;
					const T a2 = a * a;
					const T term = std::sqrt((r2 - a2) * (r2 - a2) + 4 * a2 * z * z);
					const T r = std::sqrt(0.5 * (r2 - a2 + term));
					return r;
				}

				void compute_kerr_schild(const morpheus::Vector<T>& X, morpheus::Tensor<T, 2>& g) const {
					assert(X.size() == 4);
					const T x = X(1), y = X(2), z = X(3);

					const T r = kerr_schild_radius(x, y, z);
					const T cos_theta = (r > 1e-14) ? z / r : 0.0;
					const T denom = r * r + a * a * cos_theta * cos_theta;
					const T H = (denom > 1e-14) ? (M * r) / denom : 0.0;

					const T denom_vec = r * r + a * a;
					T lx = (denom_vec > 1e-14) ? (r * x + a * y) / denom_vec : 0.0;
					T ly = (denom_vec > 1e-14) ? (r * y - a * x) / denom_vec : 0.0;
					T lz = (r > 1e-14) ? z / r : 0.0;

					T norm_l = std::sqrt(lx * lx + ly * ly + lz * lz);
					if (norm_l > 1e-14) {
						lx /= norm_l;
						ly /= norm_l;
						lz /= norm_l;
					}

					g.resize({4, 4});
					g.fill(0);

					g(0, 0) = -1.0;
					g(1, 1) = 1.0;
					g(2, 2) = 1.0;
					g(3, 3) = 1.0;

					const std::array<T, 4> l = {1.0, lx, ly, lz};

					for (size_t mu = 0; mu < 4; ++mu)
						for (size_t nu = 0; nu < 4; ++nu)
							g(mu, nu) += 2.0 * H * l[mu] * l[nu];
				}
		};
} 
