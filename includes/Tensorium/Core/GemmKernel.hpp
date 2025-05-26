#pragma once

#include "Matrix.hpp"

namespace tensorium {
	template<typename K>
		class GemmKernel {
			public:
				using Simd = simd::SimdTraits<K, DefaultISA>;
				using reg = typename Simd::reg;
				static constexpr int TileRows = Simd::width * 2;
				static constexpr int TileCols = 6;
				static constexpr int NThreads = 16;
				static constexpr int BlockRows = TileRows * NThreads * 5;
				static constexpr int BlockCols = TileCols * NThreads * 50;
				static constexpr int BlockDepth = 500;

				static constexpr int8_t mask[32] __attribute__((aligned(64))) = {
					-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
				};

				struct FmaContext {
					const K* a;
					const K* b;
					int kc;

					reg a0 = Simd::setzero();
					reg a1 = Simd::setzero();
					reg bcast = Simd::setzero();

					reg accum[2 * TileCols] = {}; 

					reg& acc(int row, int col) {
						return accum[2 * col + row];
					}

					void advance() {
						a += TileRows;
						b += TileCols;
					}
				};

				static inline int min_int(int x, int y) { return (x < y) ? x : y; }

				static inline void packTileB(
						const K* B,
						K* blockB_packed,
						int nr,
						int kc,
						int ldb
						) {
					for (int p = 0; p < kc; ++p) {
						for (int j = 0; j < nr; ++j) {
							blockB_packed[p * TileCols + j] = B[p * ldb + j];
						}
						for (int j = nr; j < TileCols; ++j) {
							blockB_packed[p * TileCols + j] = K(0);
						}
					}
				}

				static inline void packBlockB(
						const K* B,
						K* blockB_packed,
						int nc,
						int kc,
						int ldb 
						) {
#pragma omp parallel for num_threads(NThreads)
					for (int col = 0; col < nc; col += TileCols) {
						int nr = std::min(TileCols, nc - col);
						K* panel_ptr = blockB_packed + (col / TileCols) * (kc * TileCols);
						packTileB(
								B + col, panel_ptr,
								nr, kc, ldb
								);
					}
				}

				static inline void packTileA(
						const K* A,
						K* blockA_packed,
						int mr,
						int kc,
						int lda
						) {
					for (int p = 0; p < kc; ++p) {
						for (int i = 0; i < mr; ++i) {
							blockA_packed[i] = A[i * lda + p];
						}
						for (int i = mr; i < TileRows; ++i) {
							blockA_packed[i] = K(0);
						}
						blockA_packed += TileRows;
					}
				}

				static inline void packBlockA(
						const K* A,
						K* blockA_packed,
						int mc,
						int kc,
						int lda 
						) {
#pragma omp parallel for num_threads(NThreads)
					for (int i = 0; i < mc; i += TileRows) {
						int mr = std::min(TileRows, mc - i);
						int tile_idx = i / TileRows;
						packTileA(
								A + i * lda,
								blockA_packed + tile_idx * (kc * TileRows),
								mr, kc, lda
								);
					}
				}
				static inline void build_masks(__m256i* m0, __m256i* m1, int mr)
				{
					alignas(32) int32_t mask_data[16] = {0};
					for (int i = 0; i < mr; ++i) mask_data[i] = -1;
					*m0 = _mm256_load_si256((__m256i*)mask_data);
					*m1 = _mm256_load_si256((__m256i*)(mask_data + 8));
				}

				static inline void kernel_16x6_zero_init_accum(
						K* blockA_packed,
						K* blockB_packed,
						K* C,
						int mr,
						int nr,
						int kc,
						int ldc
						) {
					constexpr int W = Simd::width;
					reg accum[TileCols][2] = {};
					__m256i mask0{}, mask1{};
					if (mr != TileRows) build_masks(&mask0, &mask1, mr);

					for (int p = 0; p < kc; ++p) {
						reg a0 = Simd::loadu(blockA_packed + 0);
						reg a1 = Simd::loadu(blockA_packed + W);

						for (int j = 0; j < nr; ++j) {
							reg b = Simd::broadcast(blockB_packed + p * TileCols + j);
							accum[j][0] = Simd::fmadd(a0, b, accum[j][0]);
							accum[j][1] = Simd::fmadd(a1, b, accum[j][1]);
						}

						blockA_packed += TileRows;
					}

					for (int j = 0; j < nr; ++j) {
						K* Ccol = C + j * ldc;
						if (mr == TileRows) {
							Simd::storeu(Ccol + 0, accum[j][0]);
							Simd::storeu(Ccol + W, accum[j][1]);
						} else {
							Simd::maskstore(Ccol + 0, mask0, accum[j][0]);
							Simd::maskstore(Ccol + W, mask1, accum[j][1]);
						}
					}
				}

				static inline void kernel_16x6_load_accum(
						K* blockA_packed,
						K* blockB_packed,
						K* C,
						int mr,
						int nr,
						int kc,
						int ldc
						) {
					constexpr int W = Simd::width;
					reg accum[TileCols][2];
					__m256i mask0{}, mask1{};
					if (mr != TileRows) build_masks(&mask0, &mask1, mr);

					for (int j = 0; j < nr; ++j) {
						K* Ccol = C + j * ldc;
						if (mr == TileRows) {
							accum[j][0] = Simd::loadu(Ccol + 0);
							accum[j][1] = Simd::loadu(Ccol + W);
						} else {
							accum[j][0] = Simd::maskload(Ccol + 0, mask0);
							accum[j][1] = Simd::maskload(Ccol + W, mask1);
						}
					}

					for (int p = 0; p < kc; ++p) {
						reg a0 = Simd::loadu(blockA_packed + 0);
						reg a1 = Simd::loadu(blockA_packed + W);

						for (int j = 0; j < nr; ++j) {
							reg b = Simd::broadcast(blockB_packed + p * TileCols + j);
							accum[j][0] = Simd::fmadd(a0, b, accum[j][0]);
							accum[j][1] = Simd::fmadd(a1, b, accum[j][1]);
						}

						blockA_packed += TileRows;
					}

					for (int j = 0; j < nr; ++j) {
						K* Ccol = C + j * ldc;
						if (mr == TileRows) {
							Simd::storeu(Ccol + 0, accum[j][0]);
							Simd::storeu(Ccol + W, accum[j][1]);
						} else {
							Simd::maskstore(Ccol + 0, mask0, accum[j][0]);
							Simd::maskstore(Ccol + W, mask1, accum[j][1]);
						}
					}
				}
				inline void matmul_parallel(
						K* A, K* B, K* C,
						int M, int N, int Kdim
						) {
					constexpr int MC = BlockRows;
					constexpr int NC = BlockCols;
					constexpr int KC = BlockDepth;
					constexpr int MR = TileRows;
					constexpr int NR = TileCols;

					K* blockA_packed = nullptr;
					K* blockB_packed = nullptr;
					posix_memalign((void**)&blockA_packed, 64, sizeof(K) * MC * KC);
					posix_memalign((void**)&blockB_packed, 64, sizeof(K) * NC * KC);
					for (int jc = 0; jc < N; jc += NC) {
						int nc = std::min(NC, N - jc);

						for (int pc = 0; pc < Kdim; pc += KC) {
							int kc = std::min(KC, Kdim - pc);
							packBlockB(
									B + jc * Kdim + pc, 
									blockB_packed,
									nc, kc, Kdim
									);
							for (int ic = 0; ic < M; ic += MC) {
								int mc = std::min(MC, M - ic);
								packBlockA(
										A + ic * Kdim + pc,  
										blockA_packed,
										mc, kc, Kdim
										);

#pragma omp parallel for collapse(2)
								for (int jr = 0; jr < nc; jr += NR) {
									for (int ir = 0; ir < mc; ir += MR) {
										int nr = std::min(NR, nc - jr);
										int mr = std::min(MR, mc - ir);

										K* Ctile = C + (ic + ir) * N + (jc + jr);

										if (pc == 0) {
											kernel_16x6_zero_init_accum(
													blockA_packed + ir * kc,
													blockB_packed + jr * kc,
													Ctile, mr, nr, kc, N);
										} else {
											kernel_16x6_load_accum(
													blockA_packed + ir * kc,
													blockB_packed + jr * kc,
													Ctile, mr, nr, kc, N);

										}
									}
								}
							}
						}
					}	
					free(blockA_packed);
					free(blockB_packed);
				}
		};
} // namespace tensorium
