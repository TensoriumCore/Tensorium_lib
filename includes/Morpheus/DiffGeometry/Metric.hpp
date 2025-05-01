#pragma once

#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "../Core/Tensor.hpp"
#include "../Core/Vector.hpp"

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
					} else {
						throw std::invalid_argument("Unknown metric type: " + type);
					}
				}

			private:
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

					g.resize(4, 4);
					g.fill(T(0));

					g(0, 0) = -(T(1) - (T(2) * M * r) / Sigma);
					g(0, 3) = g(3, 0) = -T(2) * M * r * a * sin2 / Sigma;
					g(1, 1) = Sigma / Delta;
					g(2, 2) = Sigma;
					g(3, 3) = (r * r + a * a + T(2) * M * r * a * a * sin2 / Sigma) * sin2;
				}
		};
} 
