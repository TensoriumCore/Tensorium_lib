#pragma once

#include "../Core/Vector.hpp"
#include "../Core/Tensor.hpp"
#include "../Core/Matrix.hpp"
#include "../Functionnal/FunctionnalRG.hpp"
#include "../DiffGeometry/ChristoffelSymbol.hpp"
#include "../DiffGeometry/Metric.hpp"
#include "../Core/Derivate.hpp"
namespace morpheus_RG {
	template <typename T>
		class RicciTensor {
			public:
				using Tensor4D = morpheus::Tensor<T, 4>;
				using Tensor2D = morpheus::Tensor<T, 2>;
				using VectorT  = morpheus::Vector<T>;

				static morpheus::Tensor<T, 2> contract_to_ricci(const Tensor4D& Riemann, const morpheus::Tensor<T, 2>& g_inv) {
					constexpr size_t dim = 4;
					morpheus::Tensor<T, 2> Ricci({dim, dim});
					Ricci.fill(T(0));

					for (size_t mu = 0; mu < dim; ++mu) {
						for (size_t nu = 0; nu < dim; ++nu) {
							for (size_t rho = 0; rho < dim; ++rho) {
								for (size_t sigma = 0; sigma < dim; ++sigma) {
									Ricci(mu, nu) += g_inv(rho, sigma) * Riemann(rho, sigma, mu, nu);
								}
							}
						}
					}

					return Ricci;
				}
			
				static double compute_ricci_scalar(const Tensor2D& Ricci, const morpheus::Tensor<T, 2>& g_inv) {
					constexpr size_t dim = 4;
					double R = 0.0;
					for (size_t mu = 0; mu < dim; ++mu) {
						for (size_t nu = 0; nu < dim; ++nu) {
							R += g_inv(mu, nu) * Ricci(mu, nu);
						}
					}
					return R;
				}

				static void print_componentwise(const Tensor2D& R, T threshold = 1e-12) {
					constexpr size_t dim = 4;
					std::cout << "\nRicci tensor components:\n";
					for (size_t mu = 0; mu < dim; ++mu) {
						for (size_t nu = 0; nu < dim; ++nu) {
							T val = R(mu, nu);
							if (std::abs(val) < threshold)
								val = 0.0;
							std::cout << std::setw(10) << std::fixed << std::setprecision(6) << val << " ";
						}
						std::cout << "\n";
					}
				}
			
					
				static void print(const Tensor2D& R, const std::string& name = "Ricci") {
					constexpr size_t dim = 4;
					std::cout << name << " tensor components:\n";
					for (size_t mu = 0; mu < dim; ++mu)
						for (size_t nu = 0; nu < dim; ++nu) {
							const T val = R(mu, nu);
							if (std::abs(val) > T(1e-12)) {
								std::cout << name << "[" << mu << "][" << nu << "] = " << val << "\n";
							}
						}
				}

				static void print(const Tensor2D& R, const morpheus::Tensor<T, 2>& g_inv, const std::string& name = "Ricci") {
					constexpr size_t dim = 4;
					std::cout << name << " tensor components:\n";
					for (size_t mu = 0; mu < dim; ++mu)
						for (size_t nu = 0; nu < dim; ++nu) {
							const T val = R(mu, nu);
							if (std::abs(val) > T(1e-12)) {
								std::cout << name << "[" << mu << "][" << nu << "] = " << val << "\n";
							}
						}
				}

				static void print_ricci_scalar(const Tensor2D& Ricci, const morpheus::Tensor<T, 2>& g_inv) {
					constexpr size_t dim = 4;
					double R = compute_ricci_scalar(Ricci, g_inv);
					std::cout << "Ricci scalar R = " << R << "\n";
				}
		};
}


