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
#include <cassert>
#include <numeric>


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

				Derivate(const Matrix<K>& m)
					: rows(m.rows), cols(m.cols), data(m.data), block_size(detect_optimal_block_size()) {}

				K& operator()(size_t i, size_t j) {
					return data[i * cols + j];
				}

				const K& operator()(size_t i, size_t j) const {
					return data[i * cols + j];
				}

				size_t size() const {
					return rows * cols;
				}

				__attribute__((always_inline, hot, flatten))
					inline void centered_derivative(const Derivate<K>& input, Derivate<K>& output, size_t axis, K dx) const {
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg  = typename Simd::reg;
						const size_t simd_width = Simd::width;

						const K inv_2dx = K(1) / (K(2) * dx);

						if (axis == 1) { 
#pragma omp parallel for
							for (size_t i = 0; i < input.rows; ++i) {
								size_t j = 1;

								for (; j + simd_width <= input.cols - 1; j += simd_width) {
									const K* left_ptr  = &input(i, j - 1);
									const K* right_ptr = &input(i, j + 1);
									K* out_ptr         = &output(i, j);

									reg left  = Simd::loadu(left_ptr);
									reg right = Simd::loadu(right_ptr);
									reg diff  = Simd::sub(right, left);
									reg res   = Simd::mul(diff, Simd::set1(inv_2dx));
									Simd::store(out_ptr, res);
								}

								for (; j < input.cols - 1; ++j) {
									output(i, j) = (input(i, j + 1) - input(i, j - 1)) * inv_2dx;
								}

								output(i, 0) = K(0);
								output(i, input.cols - 1) = K(0);
							}
						}

						else if (axis == 0) {
#pragma omp parallel for
							for (size_t i = 1; i < input.rows - 1; ++i) {
								for (size_t j = 0; j < input.cols; ++j) {
									output(i, j) = (input(i + 1, j) - input(i - 1, j)) * inv_2dx;
								}
							}
							for (size_t j = 0; j < input.cols; ++j) {
								output(0, j) = K(0);
								output(input.rows - 1, j) = K(0);
							}
						}

						else {
							std::cerr << "Invalid axis: must be 0 or 1.\n";
						}		
					}
		};



	template<typename K, size_t Rank>
		class DerivateND {
			public:
				std::array<size_t, Rank> shape;
				aligned_vector<K> data;
				size_t block_size;

				DerivateND(const std::array<size_t, Rank>& dims) 
					: shape(dims),
					data(std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>()), K()),
					block_size(detect_optimal_block_size()) 
			{
				std::cout << "Auto-selected BLOCK_SIZE = " << block_size << std::endl;
			}

				inline size_t flatten_index(const std::array<size_t, Rank>& indices) const {
					size_t index = 0, stride = 1;
					for (int i = Rank - 1; i >= 0; --i) {
						index += indices[i] * stride;
						stride *= shape[i];
					}
					return index;
				}

				inline K& operator()(const std::array<size_t, Rank>& indices) {
					return data[flatten_index(indices)];
				}

				inline const K& operator()(const std::array<size_t, Rank>& indices) const {
					return data[flatten_index(indices)];
				}

				inline size_t size() const {
					return data.size();
				}

				__attribute__((always_inline, hot, flatten))
					inline void centered_derivative(const DerivateND<K, Rank>& input, DerivateND<K, Rank>& output, size_t axis, K dx) const {
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg  = typename Simd::reg;
						const size_t simd_width = Simd::width;

						const auto& shape = input.shape;
						const size_t total = input.size();
						const K inv_2dx = K(1) / (K(2) * dx);
						const reg inv2dx = Simd::set1(inv_2dx);

						std::array<size_t, Rank> strides;
						strides[Rank - 1] = 1;
						for (int i = Rank - 2; i >= 0; --i)
							strides[i] = strides[i + 1] * shape[i + 1];

						const size_t stride_axis = strides[axis];
						const size_t dim_axis = shape[axis];

#pragma omp parallel for schedule(static)
						for (size_t flat = 0; flat < total; flat += simd_width) {
							bool safe = true;

							for (size_t offset = 0; offset < simd_width; ++offset) {
								if (flat + offset >= total) { safe = false; break; }

								const size_t coord_axis = ((flat + offset) / stride_axis) % dim_axis;
								if (coord_axis == 0 || coord_axis >= dim_axis - 1) {
									safe = false;
									break;
								}
							}

							if (!safe) {
								for (size_t offset = 0; offset < simd_width && flat + offset < total; ++offset) {
									const size_t f = flat + offset;
									const size_t coord_axis = (f / stride_axis) % dim_axis;

									if (coord_axis == 0 || coord_axis >= dim_axis - 1) {
										output.data[f] = K(0);
										continue;
									}
									output.data[f] = (input.data[f + stride_axis] - input.data[f - stride_axis]) * inv_2dx;
								}
								continue;
							}

							const K* ptr_fwd  = input.data.data() + flat + stride_axis;
							const K* ptr_back = input.data.data() + flat - stride_axis;
							K* out_ptr        = output.data.data() + flat;

							reg forward = Simd::loadu(ptr_fwd);
							reg backward = Simd::loadu(ptr_back);
							reg diff = Simd::sub(forward, backward);
							reg result = Simd::mul(diff, inv2dx);
							Simd::store(out_ptr, result);
						}
					}
		};

}


