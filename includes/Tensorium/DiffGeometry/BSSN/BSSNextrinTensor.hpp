#pragma once

#include "../Metric.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../../Core/Derivate.hpp"
namespace tensorium {

	template<typename K>
		class ExtrinsicCurvature {
			public:
				using Vec = Vector<K>;
				using Mat = Tensor<K, 2>;

				ExtrinsicCurvature(const Metric<K>& metric) : metric_(metric) {}


				Mat compute_K_tensor(const Vec& X, K dx) const {
					Mat gamma; Vec beta; K alpha;
					metric_.BSSN(X, alpha, beta, gamma);

					Mat dgdt(3, 3);
					dgdt.fill(K(0));

					for (size_t i = 0; i < 3; ++i) {
						for (size_t j = 0; j < 3; ++j) {
							K sum = K(0);
							for (size_t k = 0; k < 3; ++k) {
								Vec Xp = X, Xm = X;
								Xp(k+1) += dx;  
								Xm(k+1) -= dx;
								Tensor<K,2> gp, gm;
								metric_.BSSN(Xp, alpha, beta, gp);
								metric_.BSSN(Xm, alpha, beta, gm);
								K dgamma_dk = (gp(i,j) - gm(i,j)) / (2*dx);

								sum += beta(k) * dgamma_dk;
							}
							dgdt(i,j) = sum;
						}
					}

					Mat Kij(3,3);
					for (size_t i = 0; i < 3; ++i)
						for (size_t j = 0; j < 3; ++j)
							Kij(i,j) = -0.5/alpha * dgdt(i,j);

					return Kij;
				}

				K compute_K_scalar(const Vec& X, K dx) const {
					Mat Kij = compute_K_tensor(X, dx);
					K trace = K(0);
					for (size_t i = 0; i < 3; ++i)
						trace += Kij(i, i);
					return trace;
				}


				// Ã_{ij} = K_{ij} − (1/3) γ_{ij} K
				Mat compute_Atilde_tensor(const Vec& X, K dx) const {
					auto Kij = compute_K_tensor(X, dx);
					auto Kscalar = compute_K_scalar(X, dx);

					Mat gamma; Vec beta; K alpha;
					metric_.BSSN(X, alpha, beta, gamma);

					Mat Atilde(3, 3);
					for (size_t i = 0; i < 3; ++i)
						for (size_t j = 0; j < 3; ++j)
							Atilde(i, j) = Kij(i, j) - (Kscalar / 3.0) * gamma(i, j);

					return Atilde;
				}

			private:
				const Metric<K>& metric_;

				Mat invert(const Mat& gamma) const {
					Mat inv(3, 3);
					const K det =
						gamma(0, 0) * (gamma(1, 1) * gamma(2, 2) - gamma(1, 2) * gamma(2, 1)) -
						gamma(0, 1) * (gamma(1, 0) * gamma(2, 2) - gamma(1, 2) * gamma(2, 0)) +
						gamma(0, 2) * (gamma(1, 0) * gamma(2, 1) - gamma(1, 1) * gamma(2, 0));

					inv(0, 0) = (gamma(1, 1) * gamma(2, 2) - gamma(1, 2) * gamma(2, 1)) / det;
					inv(0, 1) = (gamma(0, 2) * gamma(2, 1) - gamma(0, 1) * gamma(2, 2)) / det;
					inv(0, 2) = (gamma(0, 1) * gamma(1, 2) - gamma(0, 2) * gamma(1, 1)) / det;

					inv(1, 0) = (gamma(1, 2) * gamma(2, 0) - gamma(1, 0) * gamma(2, 2)) / det;
					inv(1, 1) = (gamma(0, 0) * gamma(2, 2) - gamma(0, 2) * gamma(2, 0)) / det;
					inv(1, 2) = (gamma(0, 2) * gamma(1, 0) - gamma(0, 0) * gamma(1, 2)) / det;

					inv(2, 0) = (gamma(1, 0) * gamma(2, 1) - gamma(1, 1) * gamma(2, 0)) / det;
					inv(2, 1) = (gamma(0, 1) * gamma(2, 0) - gamma(0, 0) * gamma(2, 1)) / det;
					inv(2, 2) = (gamma(0, 0) * gamma(1, 1) - gamma(0, 1) * gamma(1, 0)) / det;

					return inv;
				}
		};

}
