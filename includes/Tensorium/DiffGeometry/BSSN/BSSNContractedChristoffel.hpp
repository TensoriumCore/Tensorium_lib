#pragma once

#include "../Metric.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {

	template<typename T>
		class BSSNContractedGamma {
			public:
				/**
				 * @brief Compute the contracted Christoffel symbol \f$ \Gamma^i_{ij} = -\frac{3}{2} \partial_j \ln \chi \f$
				 *
				 * This uses the relation:
				 * \f[
				 * \Gamma^i_{ij} = -\frac{3}{2} \frac{\partial_j \chi}{\chi}
				 * \f]
				 *
				 * @param X     Spatial coordinates
				 * @param metric Metric object implementing `BSSN(X, alpha, beta, gamma)`
				 * @param dx,dy,dz Grid spacing
				 * @param chi   Conformal factor \f$ \chi \f$
				 * @return Vector containing \f$ \Gamma^i_{ij} \f$ for each j
				 */
				static tensorium::Vector<T> compute(
						const tensorium::Vector<T>& X,
						const tensorium_RG::Metric<T>& metric,
						T dx, T dy, T dz,
						T chi)
				{
					tensorium::Vector<T> d_chi(3);

					tensorium_RG::compute_partial_derivatives_scalar<T>(
							X, dx, dy, dz,
							[&](const tensorium::Vector<T>& Xs) {
							T a_tmp;
							tensorium::Vector<T> b_tmp(3);
							tensorium::Tensor<T, 2> g_tmp({3, 3});
							metric.BSSN(Xs, a_tmp, b_tmp, g_tmp);
							return compute_conformal_factor(metric, g_tmp);
							},
							d_chi
							);

					tensorium::Vector<T> out(3);
					for (size_t j = 0; j < 3; ++j)
						out[j] = -1.5 * d_chi[j] / chi;

					return out;
				}
		};

} // namespace tensorium_RG
