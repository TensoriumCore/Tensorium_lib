#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"

namespace morpheus {
	template<typename K, std::size_t Rank>
		class Tensor {
			public:
				using value_type = K;
				std::array<size_t, Rank> dimensions;
				size_t total_size;
				aligned_vector<K> data;
				size_t block_size;
				std::array<size_t, Rank> strides;

				Tensor() : total_size(0), block_size(128) {}

				Tensor(const std::array<size_t, Rank>& dims)
					: dimensions(dims), total_size(1), block_size(128) {
						strides[Rank - 1] = 1;
						for (int64_t i = Rank - 2; i >= 0; --i) {
							strides[i] = strides[i + 1] * dimensions[i + 1];
						}

						size_t total = 1;
						for (size_t i = 0; i < Rank; ++i)
							total *= dimensions[i];
						data.resize(total);
						total_size = total;
					}

				K& operator()(const std::array<size_t, Rank>& indices) {
					size_t index = flatten_index(indices);
					assert(index < total_size);
					return data[index];
				}

				const K& operator()(const std::array<size_t, Rank>& indices) const {
					size_t index = flatten_index(indices);
					assert(index < total_size);
					return data[index];
				}

				void fill(K value) {
					std::fill(data.begin(), data.end(), value);
				}

				void print_shape() const {
					std::cout << "Tensor shape: (";
					for (size_t i = 0; i < Rank; ++i) {
						std::cout << dimensions[i];
						if (i + 1 < Rank) std::cout << ", ";
					}
					std::cout << ")\n";
				}


				__attribute__((always_inline, hot, flatten))
					inline size_t flatten_index_simd(const size_t* indices, const size_t* strides) const {
						using Simd = simd::SimdTraits<size_t, DefaultISA>;
						using reg = typename Simd::reg;
						constexpr size_t W = Simd::width / sizeof(size_t);

						size_t acc = 0;
						size_t i = 0;

						for (; i + W - 1 < Rank; i += W) {
							reg idx = Simd::load(&indices[i]);
							reg str = Simd::load(&strides[i]);
							reg prod = Simd::mul(idx, str);
							acc += detail::reduce_sum(prod);
						}
						for (; i < Rank; ++i)
							acc += indices[i] * strides[i];

						return acc;
					}

				__attribute__((always_inline, hot, flatten))
					inline size_t flatten_index(const std::array<size_t, Rank>& indices) const {
						return flatten_index_simd(indices.data(), strides.data());
					}

				__attribute__((always_inline, hot, flatten))
					Tensor<K, Rank - 2> contract_simd(const Tensor<K, Rank>& t, size_t i, size_t j) const {
						static_assert(Rank >= 2, "Cannot contract tensor of rank < 2");
						assert(i < Rank && j < Rank && i != j);
						assert(t.dimensions[i] == t.dimensions[j]);

						using SimdIndex = simd::SimdTraits<size_t, DefaultISA>;
						using SimdValue = simd::SimdTraits<K, DefaultISA>;
						using reg = typename SimdValue::reg;
						constexpr size_t W = SimdValue::width;

						std::array<size_t, Rank - 2> new_dims;
						size_t d_idx = 0;
						for (size_t d = 0; d < Rank; ++d) {
							if (d != i && d != j)
								new_dims[d_idx++] = t.dimensions[d];
						}
						Tensor<K, Rank - 2> result(new_dims);

						std::array<size_t, Rank> indices{};
						std::array<size_t, Rank - 2> reduced_idx{};

						for (size_t flat = 0; flat < result.data.size(); ++flat) {
							size_t tmp = flat;
							d_idx = 0;
							for (size_t d = 0; d < Rank; ++d) {
								if (d == i || d == j) {
									indices[d] = 0;
								} else {
									indices[d] = tmp % result.dimensions[d_idx];
									tmp /= result.dimensions[d_idx++];
								}
							}

							const size_t dim = t.dimensions[i];
							size_t k = 0;
							reg acc = SimdValue::zero();

							for (; k + W - 1 < dim; k += W) {
								alignas(64) size_t k_vec[W];
								for (size_t w = 0; w < W; ++w) {
									k_vec[w] = k + w;
									indices[i] = k_vec[w];
									indices[j] = k_vec[w];
								}

								alignas(64) K vals[W];
								for (size_t w = 0; w < W; ++w)
									vals[w] = t(indices);

								reg vec = SimdValue::load(vals);
								acc = SimdValue::add(acc, vec);
							}

							K sum = detail::reduce_sum(acc);
							for (; k < dim; ++k) {
								indices[i] = k;
								indices[j] = k;
								sum += t(indices);
							}

							d_idx = 0;
							for (size_t d = 0; d < Rank; ++d) {
								if (d != i && d != j)
									reduced_idx[d_idx++] = indices[d];
							}
							result(reduced_idx) = sum;
						}

						return result;
					}

				template <size_t I, size_t J>
					Tensor<K, Rank - 2> contract_tensor() const;

					__attribute__((always_inline, hot, flatten))
					Tensor<K, 2> transpose_simd() const {
						const size_t rows = dimensions[0];
						const size_t cols = dimensions[1];

						Tensor<K, 2> result({cols, rows});

						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						constexpr size_t W = Simd::width / sizeof(K);

						for (size_t i = 0; i < rows; ++i) {
							size_t j = 0;
							for (; j + W - 1 < cols; j += W) {
								reg vec = Simd::load(&(*this)({i, j}));

								alignas(64) K temp[W];
								Simd::store(temp, vec);
								for (size_t k = 0; k < W; ++k)
									result({j + k, i}) = temp[k];
							}
							for (; j < cols; ++j)
								result({j, i}) = (*this)({i, j});
						}

						return result;
					}
			
			

					template<size_t R1, size_t R2>
						static inline Tensor<K, R1 + R2> tensor_product(const Tensor<K, R1>& A, const Tensor<K, R2>& B) {
							using Simd = simd::SimdTraits<K, DefaultISA>;
							using reg = typename Simd::reg;
							constexpr size_t W = Simd::width / sizeof(K);
							constexpr size_t R = R1 + R2;
							constexpr size_t L3_BLOCK = 128;
							constexpr size_t L1_BLOCK = 128;

							std::array<size_t, R> shape;
							for (size_t i = 0; i < R1; ++i) 
								shape[i] = A.dimensions[i];
							for (size_t i = 0; i < R2; ++i) 
								shape[R1 + i] = B.dimensions[i];

							Tensor<K, R> result(shape);

#pragma omp parallel for collapse(2)
							for (size_t a_outer = 0; a_outer < A.total_size; a_outer += L3_BLOCK) {
								for (size_t b_outer = 0; b_outer < B.total_size; b_outer += L3_BLOCK) {
									_mm_prefetch(&A.data[a_outer + L3_BLOCK], _MM_HINT_NTA);
									_mm_prefetch(&B.data[b_outer + L3_BLOCK], _MM_HINT_NTA);
									const size_t a_outer_end = std::min(a_outer + L3_BLOCK, A.total_size);
									const size_t b_outer_end = std::min(b_outer + L3_BLOCK, B.total_size);

									for (size_t a_inner = a_outer; a_inner < a_outer_end; a_inner += L1_BLOCK) {
										for (size_t b_inner = b_outer; b_inner < b_outer_end; b_inner += L1_BLOCK) {
											_mm_prefetch(&A.data[a_inner + L1_BLOCK], _MM_HINT_T0);
											_mm_prefetch(&B.data[b_inner + L1_BLOCK], _MM_HINT_T0);
											const size_t a_end = std::min(a_inner + L1_BLOCK, a_outer_end);
											const size_t b_end = std::min(b_inner + L1_BLOCK, b_outer_end);

											for (size_t a_flat = a_inner; a_flat < a_end; ++a_flat) {
												std::array<size_t, R1> idx_A;
												size_t tmp = a_flat;
												for (ssize_t i = R1 - 1; i >= 0; --i) {
													idx_A[i] = tmp % A.dimensions[i];
													tmp /= A.dimensions[i];
												}
												K a_scalar = A(idx_A);
												reg a_vec = Simd::set1(a_scalar);

												for (size_t b_flat = b_inner; b_flat + W - 1 < b_end; b_flat += W) {
													reg b_vec = Simd::loadu(&B.data[b_flat]);
													reg c_vec = Simd::mul(a_vec, b_vec);

													for (size_t w = 0; w < W; ++w) {
														std::array<size_t, R2> idx_B;
														std::array<size_t, R> idx_C;

														size_t tmpb = b_flat + w;
														for (ssize_t i = R2 - 1; i >= 0; --i) {
															idx_B[i] = tmpb % B.dimensions[i];
															tmpb /= B.dimensions[i];
														}
#pragma unroll(R1 + R2 - 1) 
														for (size_t i = 0; i < R1; ++i)
															idx_C[i] = idx_A[i];
#pragma unroll(R1 + R2 - 1)
														for (size_t i = 0; i < R2; ++i) 
															idx_C[R1 + i] = idx_B[i];

														result(idx_C) = Simd::extract(c_vec, w);
													}
												}

												for (size_t b_flat = b_end - (b_end % W); b_flat < b_end; ++b_flat) {
													std::array<size_t, R2> idx_B;
													std::array<size_t, R> idx_C;

													size_t tmpb = b_flat;
													for (ssize_t i = R2 - 1; i >= 0; --i) {
														idx_B[i] = tmpb % B.dimensions[i];
														tmpb /= B.dimensions[i];
													}
													for (size_t i = 0; i < R1; ++i) 
														idx_C[i] = idx_A[i];
													for (size_t i = 0; i < R2; ++i) 
														idx_C[R1 + i] = idx_B[i];

													result(idx_C) = a_scalar * B(idx_B);
												}
											}
										}
									}
								}
							}

							return result;
						}
		};
}
