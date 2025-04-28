#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

namespace morpheus_RG {
	template <typename T>
		class RiemannTensor {
			public:
				RiemannTensor(std::size_t dim) : dim_(dim), data_(dim * dim * dim * dim) {}

				T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
					return data_[i * dim_ * dim_ * dim_ + j * dim_ * dim_ + k * dim_ + l];
				}

				const T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
					return data_[i * dim_ * dim_ * dim_ + j * dim_ * dim_ + k * dim_ + l];
				}

			private:
				std::size_t dim_;
				std::vector<T> data_;
		};

	template <typename T>
		void print_tensor(const RiemannTensor<T>& tensor) {
			for (std::size_t i = 0; i < tensor.dim_; ++i) {
				for (std::size_t j = 0; j < tensor.dim_; ++j) {
					for (std::size_t k = 0; k < tensor.dim_; ++k) {
						for (std::size_t l = 0; l < tensor.dim_; ++l) {
							std::cout << tensor(i, j, k, l) << " ";
						}
						std::cout << "\n";
					}
					std::cout << "\n";
				}
				std::cout << "\n";
			}
		}
}
