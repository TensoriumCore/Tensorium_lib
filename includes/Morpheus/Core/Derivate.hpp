#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include "Matrix.hpp"
#include "Tensor.hpp"
#include "Vector.hpp"
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"

namespace morpheus {
	template<typename K>
		class Derivate {
			public:
				size_t rows, cols;
				aligned_vector<K> data;
				size_t block_size;
				Derivate(size_t r, size_t c) 
					: rows(r), cols(c), data(r * c, K()), block_size(detect_optimal_block_size()) {
						std::cout << "Auto-selected BLOCK_SIZE = " << block_size << std::endl;
					}
				size_t size() const {
					return rows * cols; 
				}

				Derivate(const Matrix<K>& m) : rows(m.rows), cols(m.cols), data(m.data), block_size(detect_optimal_block_size()) {}
				K& operator()(size_t i, size_t j) { return data[i * cols + j]; }
				const K& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
		};
}
