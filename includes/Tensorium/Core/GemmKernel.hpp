#pragma once

#include "Matrix.hpp"

namespace tensorium {
	template<typename K>
		class GemmKernel {
			public:
				static constexpr int TileRows   = 16;
				static constexpr int TileCols   = 6;
				static constexpr int NThreads   = 8;
				static constexpr int BlockRows  = TileRows * NThreads * 5;
				static constexpr int BlockCols  = TileCols * NThreads * 50;
				static constexpr int BlockDepth = 500;

				alignas(64) K blockA_packed[BlockRows * BlockDepth] = {};
				alignas(64) K blockB_packed[BlockCols * BlockDepth] = {};

				static inline constexpr int8_t mask[32] = {
					0, 1, 2, 3, 4, 5, 6, 7,
					8, 9,10,11,12,13,14,15,
					0, 1, 2, 3, 4, 5, 6, 7,
					8, 9,10,11,12,13,14,15
				};
				struct FmaContext {
					const K* a;
					const K* b;
					int kc;

					__m256 a0 = _mm256_setzero_ps();
					__m256 a1 = _mm256_setzero_ps();
					__m256 bcast = _mm256_setzero_ps();

					__m256 accum[2 * TileCols] = {}; 

					__m256& acc(int row, int col) {
						return accum[2 * col + row];
					}

					void advance() {
						a += TileRows;
						b += TileCols;
					}
				};

				struct AccumulatorSet01 {
					__m256* acc00;
					__m256* acc01;
					__m256* acc10;
					__m256* acc11;
				};

				struct AccumulatorSet02 {
					__m256* acc00;
					__m256* acc01;
					__m256* acc10;
					__m256* acc11;
					__m256* acc20;
					__m256* acc21;
				};

				struct AccumulatorSet03 {
					__m256* acc00; __m256* acc01;
					__m256* acc10; __m256* acc11;
					__m256* acc20; __m256* acc21;
					__m256* acc30; __m256* acc31;
				};
				struct AccumulatorSet04 {
					__m256* acc00; __m256* acc01;
					__m256* acc10; __m256* acc11;
					__m256* acc20; __m256* acc21;
					__m256* acc30; __m256* acc31;
					__m256* acc40; __m256* acc41;
				};

				struct AccumulatorSet05 {
					__m256* acc00; __m256* acc01;
					__m256* acc10; __m256* acc11;
					__m256* acc20; __m256* acc21;
					__m256* acc30; __m256* acc31;
					__m256* acc40; __m256* acc41;
					__m256* acc50; __m256* acc51;
				};

				struct MaskedLoadParams {
					float* C;
					int M;
					__m256i mask0;
					__m256i mask1;
				};

				static inline int min_int(int x, int y) { return (x < y) ? x : y; }

				static inline void packTileB(const K* B, K* blockB_packed, int nr, int kc, int ldb) {
					for (int p = 0; p < kc; ++p) {
						for (int j = 0; j < nr; ++j)
							blockB_packed[j] = B[j * ldb + p];
						for (int j = nr; j < TileCols; ++j)
							blockB_packed[j] = 0;
						blockB_packed += TileCols;
					}
				}

				static inline void packBlockB(const K* B, K* blockB_packed, int nc, int kc, int ldb) {
#pragma omp parallel for num_threads(NThreads)
					for (int j = 0; j < nc; j += TileCols) {
						int nr = min_int(TileCols, nc - j);
						packTileB(&B[j * ldb], &blockB_packed[j * kc], nr, kc, ldb);
					}
				}

				static inline void packTileA(const K* A, K* blockA_packed, int mr, int kc, int M) {
					for (int p = 0; p < kc; ++p) {
						for (int i = 0; i < mr; ++i)
							blockA_packed[i] = A[p * M + i];
						for (int i = mr; i < TileRows; ++i)
							blockA_packed[i] = 0;
						blockA_packed += TileRows;
					}
				}

				static inline void packBlockA(const K* A, K* blockA_packed, int mc, int kc, int M) {
#pragma omp parallel for num_threads(NThreads)
					for (int i = 0; i < mc; i += TileRows) {
						int mr = min_int(TileRows, mc - i);
						packTileA(&A[i], &blockA_packed[i * kc], mr, kc, M);
					}
				}

				static inline void fma_loop_01(FmaContext& ctx) {
					for (int p = 0; p < ctx.kc; ++p) {
						ctx.a0 = _mm256_loadu_ps(ctx.a);
						ctx.a1 = _mm256_loadu_ps(ctx.a + 8);

						ctx.bcast = _mm256_broadcast_sd(ctx.b);
						ctx.acc(0, 0) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 0));
						ctx.acc(1, 0) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 0));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 1);
						ctx.acc(0, 1) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 1));
						ctx.acc(1, 1) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 1));

						ctx.advance();
					}
				}
				static inline void fma_loop_02(FmaContext& ctx) {
					for (int p = 0; p < ctx.kc; ++p) {
						ctx.a0 = _mm256_loadu_ps(ctx.a);
						ctx.a1 = _mm256_loadu_ps(ctx.a + 8);

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 0);
						ctx.acc(0, 0) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 0));
						ctx.acc(1, 0) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 0));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 1);
						ctx.acc(0, 1) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 1));
						ctx.acc(1, 1) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 1));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 2);
						ctx.acc(0, 2) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 2));
						ctx.acc(1, 2) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 2));

						ctx.advance();
					}
				}

				static inline void fma_loop_03(FmaContext& ctx) {
					for (int p = 0; p < ctx.kc; ++p) {
						ctx.a0 = _mm256_loadu_ps(ctx.a);
						ctx.a1 = _mm256_loadu_ps(ctx.a + 8);

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 0);
						ctx.acc(0, 0) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 0));
						ctx.acc(1, 0) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 0));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 1);
						ctx.acc(0, 1) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 1));
						ctx.acc(1, 1) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 1));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 2);
						ctx.acc(0, 2) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 2));
						ctx.acc(1, 2) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 2));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 3);
						ctx.acc(0, 3) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 3));
						ctx.acc(1, 3) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 3));

						ctx.advance();
					}
				}

				static inline void fma_loop_04(FmaContext& ctx) {
					for (int p = 0; p < ctx.kc; ++p) {
						ctx.a0 = _mm256_loadu_ps(ctx.a);
						ctx.a1 = _mm256_loadu_ps(ctx.a + 8);

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 0);
						ctx.acc(0, 0) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 0));
						ctx.acc(1, 0) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 0));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 1);
						ctx.acc(0, 1) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 1));
						ctx.acc(1, 1) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 1));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 2);
						ctx.acc(0, 2) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 2));
						ctx.acc(1, 2) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 2));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 3);
						ctx.acc(0, 3) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 3));
						ctx.acc(1, 3) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 3));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 4);
						ctx.acc(0, 4) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 4));
						ctx.acc(1, 4) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 4));

						ctx.advance();
					}
				}

				static inline void fma_loop_05(FmaContext& ctx) {
					for (int p = 0; p < ctx.kc; ++p) {
						ctx.a0 = _mm256_loadu_ps(ctx.a);
						ctx.a1 = _mm256_loadu_ps(ctx.a + 8);

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 0);
						ctx.acc(0, 0) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 0));
						ctx.acc(1, 0) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 0));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 1);
						ctx.acc(0, 1) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 1));
						ctx.acc(1, 1) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 1));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 2);
						ctx.acc(0, 2) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 2));
						ctx.acc(1, 2) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 2));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 3);
						ctx.acc(0, 3) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 3));
						ctx.acc(1, 3) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 3));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 4);
						ctx.acc(0, 4) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 4));
						ctx.acc(1, 4) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 4));

						ctx.bcast = _mm256_broadcast_sd(ctx.b + 5);
						ctx.acc(0, 5) = _mm256_fmadd_ps(ctx.a0, ctx.bcast, ctx.acc(0, 5));
						ctx.acc(1, 5) = _mm256_fmadd_ps(ctx.a1, ctx.bcast, ctx.acc(1, 5));

						ctx.advance();
					}
				}

				static inline void build_masks(__m256i* packed_mask_0, __m256i* packed_mask_1, int mr) {
					*packed_mask_0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr]));
					*packed_mask_1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr + 8]));
				}
				static inline void maskload_accum_01(const MaskedLoadParams& p, const AccumulatorSet01& acc) {
					*acc.acc00 = _mm256_maskload_ps(p.C, p.mask0);
					*acc.acc01 = _mm256_maskload_ps(p.C + 8, p.mask1);
					*acc.acc10 = _mm256_maskload_ps(p.C + p.M, p.mask0);
					*acc.acc11 = _mm256_maskload_ps(p.C + p.M + 8, p.mask1);
				}

				static inline void maskload_accum_02(const MaskedLoadParams& p, const AccumulatorSet02& acc) {
					*acc.acc00 = _mm256_maskload_ps(p.C, p.mask0);
					*acc.acc01 = _mm256_maskload_ps(p.C + 8, p.mask1);
					*acc.acc10 = _mm256_maskload_ps(p.C + p.M, p.mask0);
					*acc.acc11 = _mm256_maskload_ps(p.C + p.M + 8, p.mask1);
					*acc.acc20 = _mm256_maskload_ps(p.C + 2 * p.M, p.mask0);
					*acc.acc21 = _mm256_maskload_ps(p.C + 2 * p.M + 8, p.mask1);
				}

				static inline void maskload_accum_03(const MaskedLoadParams& p, const AccumulatorSet03& acc) {
					*acc.acc00 = _mm256_maskload_ps(p.C, p.mask0);
					*acc.acc01 = _mm256_maskload_ps(p.C + 8, p.mask1);
					*acc.acc10 = _mm256_maskload_ps(p.C + p.M, p.mask0);
					*acc.acc11 = _mm256_maskload_ps(p.C + p.M + 8, p.mask1);
					*acc.acc20 = _mm256_maskload_ps(p.C + 2 * p.M, p.mask0);
					*acc.acc21 = _mm256_maskload_ps(p.C + 2 * p.M + 8, p.mask1);
					*acc.acc30 = _mm256_maskload_ps(p.C + 3 * p.M, p.mask0);
					*acc.acc31 = _mm256_maskload_ps(p.C + 3 * p.M + 8, p.mask1);
				}

				static inline void maskload_accum_04(const MaskedLoadParams& p, const AccumulatorSet04& acc) {
					*acc.acc00 = _mm256_maskload_ps(p.C, p.mask0);
					*acc.acc01 = _mm256_maskload_ps(p.C + 8, p.mask1);
					*acc.acc10 = _mm256_maskload_ps(p.C + p.M, p.mask0);
					*acc.acc11 = _mm256_maskload_ps(p.C + p.M + 8, p.mask1);
					*acc.acc20 = _mm256_maskload_ps(p.C + 2 * p.M, p.mask0);
					*acc.acc21 = _mm256_maskload_ps(p.C + 2 * p.M + 8, p.mask1);
					*acc.acc30 = _mm256_maskload_ps(p.C + 3 * p.M, p.mask0);
					*acc.acc31 = _mm256_maskload_ps(p.C + 3 * p.M + 8, p.mask1);
					*acc.acc40 = _mm256_maskload_ps(p.C + 4 * p.M, p.mask0);
					*acc.acc41 = _mm256_maskload_ps(p.C + 4 * p.M + 8, p.mask1);
				}

				static inline void maskload_accum_05(const MaskedLoadParams& p, const AccumulatorSet05& acc) {
					*acc.acc00 = _mm256_maskload_ps(p.C, p.mask0);
					*acc.acc01 = _mm256_maskload_ps(p.C + 8, p.mask1);
					*acc.acc10 = _mm256_maskload_ps(p.C + p.M, p.mask0);
					*acc.acc11 = _mm256_maskload_ps(p.C + p.M + 8, p.mask1);
					*acc.acc20 = _mm256_maskload_ps(p.C + 2 * p.M, p.mask0);
					*acc.acc21 = _mm256_maskload_ps(p.C + 2 * p.M + 8, p.mask1);
					*acc.acc30 = _mm256_maskload_ps(p.C + 3 * p.M, p.mask0);
					*acc.acc31 = _mm256_maskload_ps(p.C + 3 * p.M + 8, p.mask1);
					*acc.acc40 = _mm256_maskload_ps(p.C + 4 * p.M, p.mask0);
					*acc.acc41 = _mm256_maskload_ps(p.C + 4 * p.M + 8, p.mask1);
					*acc.acc50 = _mm256_maskload_ps(p.C + 5 * p.M, p.mask0);
					*acc.acc51 = _mm256_maskload_ps(p.C + 5 * p.M + 8, p.mask1);
				}

				struct LoadParams {
					float* C;
					int M;
				};

				static inline void load_accum_00(const LoadParams& p, const AccumulatorSet01& acc) {
					*acc.acc00 = _mm256_loadu_ps(p.C);
					*acc.acc01 = _mm256_loadu_ps(p.C + 8);
				}

				static inline void load_accum_01(const LoadParams& p, const AccumulatorSet01& acc) {
					*acc.acc00 = _mm256_loadu_ps(p.C);
					*acc.acc01 = _mm256_loadu_ps(p.C + 8);
					*acc.acc10 = _mm256_loadu_ps(p.C + p.M);
					*acc.acc11 = _mm256_loadu_ps(p.C + p.M + 8);
				}

				static inline void load_accum_02(const LoadParams& p, const AccumulatorSet02& acc) {
					*acc.acc00 = _mm256_loadu_ps(p.C);
					*acc.acc01 = _mm256_loadu_ps(p.C + 8);
					*acc.acc10 = _mm256_loadu_ps(p.C + p.M);
					*acc.acc11 = _mm256_loadu_ps(p.C + p.M + 8);
					*acc.acc20 = _mm256_loadu_ps(p.C + 2 * p.M);
					*acc.acc21 = _mm256_loadu_ps(p.C + 2 * p.M + 8);
				}

				static inline void load_accum_03(const LoadParams& p, const AccumulatorSet03& acc) {
					*acc.acc00 = _mm256_loadu_ps(p.C);
					*acc.acc01 = _mm256_loadu_ps(p.C + 8);
					*acc.acc10 = _mm256_loadu_ps(p.C + p.M);
					*acc.acc11 = _mm256_loadu_ps(p.C + p.M + 8);
					*acc.acc20 = _mm256_loadu_ps(p.C + 2 * p.M);
					*acc.acc21 = _mm256_loadu_ps(p.C + 2 * p.M + 8);
					*acc.acc30 = _mm256_loadu_ps(p.C + 3 * p.M);
					*acc.acc31 = _mm256_loadu_ps(p.C + 3 * p.M + 8);
				}

				static inline void load_accum_04(const LoadParams& p, const AccumulatorSet04& acc) {
					*acc.acc00 = _mm256_loadu_ps(p.C);
					*acc.acc01 = _mm256_loadu_ps(p.C + 8);
					*acc.acc10 = _mm256_loadu_ps(p.C + p.M);
					*acc.acc11 = _mm256_loadu_ps(p.C + p.M + 8);
					*acc.acc20 = _mm256_loadu_ps(p.C + 2 * p.M);
					*acc.acc21 = _mm256_loadu_ps(p.C + 2 * p.M + 8);
					*acc.acc30 = _mm256_loadu_ps(p.C + 3 * p.M);
					*acc.acc31 = _mm256_loadu_ps(p.C + 3 * p.M + 8);
					*acc.acc40 = _mm256_loadu_ps(p.C + 4 * p.M);
					*acc.acc41 = _mm256_loadu_ps(p.C + 4 * p.M + 8);
				}

				static inline void load_accum_05(const LoadParams& p, const AccumulatorSet05& acc) {
					*acc.acc00 = _mm256_loadu_ps(p.C);
					*acc.acc01 = _mm256_loadu_ps(p.C + 8);
					*acc.acc10 = _mm256_loadu_ps(p.C + p.M);
					*acc.acc11 = _mm256_loadu_ps(p.C + p.M + 8);
					*acc.acc20 = _mm256_loadu_ps(p.C + 2 * p.M);
					*acc.acc21 = _mm256_loadu_ps(p.C + 2 * p.M + 8);
					*acc.acc30 = _mm256_loadu_ps(p.C + 3 * p.M);
					*acc.acc31 = _mm256_loadu_ps(p.C + 3 * p.M + 8);
					*acc.acc40 = _mm256_loadu_ps(p.C + 4 * p.M);
					*acc.acc41 = _mm256_loadu_ps(p.C + 4 * p.M + 8);
					*acc.acc50 = _mm256_loadu_ps(p.C + 5 * p.M);
					*acc.acc51 = _mm256_loadu_ps(p.C + 5 * p.M + 8);
				}

				struct StoreParams {
					float* C;
					int M;
				};

				static inline void store_accum_01(const StoreParams& p, const AccumulatorSet01& acc) {
					_mm256_storeu_ps(p.C,         *acc.acc00);
					_mm256_storeu_ps(p.C + 8,     *acc.acc01);
					_mm256_storeu_ps(p.C + p.M,   *acc.acc10);
					_mm256_storeu_ps(p.C + p.M + 8, *acc.acc11);
				}

				static inline void store_accum_02(const StoreParams& p, const AccumulatorSet02& acc) {
					_mm256_storeu_ps(p.C,             *acc.acc00);
					_mm256_storeu_ps(p.C + 8,         *acc.acc01);
					_mm256_storeu_ps(p.C + p.M,       *acc.acc10);
					_mm256_storeu_ps(p.C + p.M + 8,   *acc.acc11);
					_mm256_storeu_ps(p.C + 2 * p.M,   *acc.acc20);
					_mm256_storeu_ps(p.C + 2 * p.M + 8, *acc.acc21);
				}

				static inline void store_accum_03(const StoreParams& p, const AccumulatorSet03& acc) {
					_mm256_storeu_ps(p.C,             *acc.acc00);
					_mm256_storeu_ps(p.C + 8,         *acc.acc01);
					_mm256_storeu_ps(p.C + p.M,       *acc.acc10);
					_mm256_storeu_ps(p.C + p.M + 8,   *acc.acc11);
					_mm256_storeu_ps(p.C + 2 * p.M,   *acc.acc20);
					_mm256_storeu_ps(p.C + 2 * p.M + 8, *acc.acc21);
					_mm256_storeu_ps(p.C + 3 * p.M,   *acc.acc30);
					_mm256_storeu_ps(p.C + 3 * p.M + 8, *acc.acc31);
				}

				static inline void store_accum_04(const StoreParams& p, const AccumulatorSet04& acc) {
					_mm256_storeu_ps(p.C,             *acc.acc00);
					_mm256_storeu_ps(p.C + 8,         *acc.acc01);
					_mm256_storeu_ps(p.C + p.M,       *acc.acc10);
					_mm256_storeu_ps(p.C + p.M + 8,   *acc.acc11);
					_mm256_storeu_ps(p.C + 2 * p.M,   *acc.acc20);
					_mm256_storeu_ps(p.C + 2 * p.M + 8, *acc.acc21);
					_mm256_storeu_ps(p.C + 3 * p.M,   *acc.acc30);
					_mm256_storeu_ps(p.C + 3 * p.M + 8, *acc.acc31);
					_mm256_storeu_ps(p.C + 4 * p.M,   *acc.acc40);
					_mm256_storeu_ps(p.C + 4 * p.M + 8, *acc.acc41);
				}

				static inline void store_accum_05(const StoreParams& p, const AccumulatorSet05& acc) {
					_mm256_storeu_ps(p.C,             *acc.acc00);
					_mm256_storeu_ps(p.C + 8,         *acc.acc01);
					_mm256_storeu_ps(p.C + p.M,       *acc.acc10);
					_mm256_storeu_ps(p.C + p.M + 8,   *acc.acc11);
					_mm256_storeu_ps(p.C + 2 * p.M,   *acc.acc20);
					_mm256_storeu_ps(p.C + 2 * p.M + 8, *acc.acc21);
					_mm256_storeu_ps(p.C + 3 * p.M,   *acc.acc30);
					_mm256_storeu_ps(p.C + 3 * p.M + 8, *acc.acc31);
					_mm256_storeu_ps(p.C + 4 * p.M,   *acc.acc40);
					_mm256_storeu_ps(p.C + 4 * p.M + 8, *acc.acc41);
					_mm256_storeu_ps(p.C + 5 * p.M,   *acc.acc50);
					_mm256_storeu_ps(p.C + 5 * p.M + 8, *acc.acc51);
				}

				struct MaskedStoreParams {
					float* C;
					int M;
					__m256i mask0;
					__m256i mask1;
				};

				static inline void maskstore_accum_01(const MaskedStoreParams& p, const AccumulatorSet01& acc) {
					_mm256_maskstore_ps(p.C,           p.mask0, *acc.acc00);
					_mm256_maskstore_ps(p.C + 8,       p.mask1, *acc.acc01);
					_mm256_maskstore_ps(p.C + p.M,     p.mask0, *acc.acc10);
					_mm256_maskstore_ps(p.C + p.M + 8, p.mask1, *acc.acc11);
				}

				static inline void maskstore_accum_02(const MaskedStoreParams& p, const AccumulatorSet02& acc) {
					_mm256_maskstore_ps(p.C,               p.mask0, *acc.acc00);
					_mm256_maskstore_ps(p.C + 8,           p.mask1, *acc.acc01);
					_mm256_maskstore_ps(p.C + p.M,         p.mask0, *acc.acc10);
					_mm256_maskstore_ps(p.C + p.M + 8,     p.mask1, *acc.acc11);
					_mm256_maskstore_ps(p.C + 2 * p.M,     p.mask0, *acc.acc20);
					_mm256_maskstore_ps(p.C + 2 * p.M + 8, p.mask1, *acc.acc21);
				}

				static inline void maskstore_accum_03(const MaskedStoreParams& p, const AccumulatorSet03& acc) {
					_mm256_maskstore_ps(p.C,               p.mask0, *acc.acc00);
					_mm256_maskstore_ps(p.C + 8,           p.mask1, *acc.acc01);
					_mm256_maskstore_ps(p.C + p.M,         p.mask0, *acc.acc10);
					_mm256_maskstore_ps(p.C + p.M + 8,     p.mask1, *acc.acc11);
					_mm256_maskstore_ps(p.C + 2 * p.M,     p.mask0, *acc.acc20);
					_mm256_maskstore_ps(p.C + 2 * p.M + 8, p.mask1, *acc.acc21);
					_mm256_maskstore_ps(p.C + 3 * p.M,     p.mask0, *acc.acc30);
					_mm256_maskstore_ps(p.C + 3 * p.M + 8, p.mask1, *acc.acc31);
				}

				static inline void maskstore_accum_04(const MaskedStoreParams& p, const AccumulatorSet04& acc) {
					_mm256_maskstore_ps(p.C,               p.mask0, *acc.acc00);
					_mm256_maskstore_ps(p.C + 8,           p.mask1, *acc.acc01);
					_mm256_maskstore_ps(p.C + p.M,         p.mask0, *acc.acc10);
					_mm256_maskstore_ps(p.C + p.M + 8,     p.mask1, *acc.acc11);
					_mm256_maskstore_ps(p.C + 2 * p.M,     p.mask0, *acc.acc20);
					_mm256_maskstore_ps(p.C + 2 * p.M + 8, p.mask1, *acc.acc21);
					_mm256_maskstore_ps(p.C + 3 * p.M,     p.mask0, *acc.acc30);
					_mm256_maskstore_ps(p.C + 3 * p.M + 8, p.mask1, *acc.acc31);
					_mm256_maskstore_ps(p.C + 4 * p.M,     p.mask0, *acc.acc40);
					_mm256_maskstore_ps(p.C + 4 * p.M + 8, p.mask1, *acc.acc41);
				}

				static inline void maskstore_accum_05(const MaskedStoreParams& p, const AccumulatorSet05& acc) {
					_mm256_maskstore_ps(p.C,               p.mask0, *acc.acc00);
					_mm256_maskstore_ps(p.C + 8,           p.mask1, *acc.acc01);
					_mm256_maskstore_ps(p.C + p.M,         p.mask0, *acc.acc10);
					_mm256_maskstore_ps(p.C + p.M + 8,     p.mask1, *acc.acc11);
					_mm256_maskstore_ps(p.C + 2 * p.M,     p.mask0, *acc.acc20);
					_mm256_maskstore_ps(p.C + 2 * p.M + 8, p.mask1, *acc.acc21);
					_mm256_maskstore_ps(p.C + 3 * p.M,     p.mask0, *acc.acc30);
					_mm256_maskstore_ps(p.C + 3 * p.M + 8, p.mask1, *acc.acc31);
					_mm256_maskstore_ps(p.C + 4 * p.M,     p.mask0, *acc.acc40);
					_mm256_maskstore_ps(p.C + 4 * p.M + 8, p.mask1, *acc.acc41);
					_mm256_maskstore_ps(p.C + 5 * p.M,     p.mask0, *acc.acc50);
					_mm256_maskstore_ps(p.C + 5 * p.M + 8, p.mask1, *acc.acc51);
				}

				static  inline void kernel_16x6_load_accum(float* blockA_packed,
						float* blockB_packed,
						float* C,
						int mr,
						int nr,
						int kc,
						int M) {
					GemmKernel::FmaContext ctx = {
						.a = blockA_packed,
						.b = blockB_packed,
						.kc = kc
					};

					__m256 a0_packFloat8 = {};
					__m256 a1_packFloat8 = {};
					__m256 b_packFloat8  = {};
					__m256i packed_mask_0 = {};
					__m256i packed_mask_1 = {};

					if (mr != 16) {
						GemmKernel::build_masks(&packed_mask_0, &packed_mask_1, mr);

						GemmKernel::MaskedLoadParams load_params = {
							.C = C,
							.M = M,
							.mask0 = packed_mask_0,
							.mask1 = packed_mask_1
						};
						GemmKernel::MaskedStoreParams store_params = load_params;

						switch (nr) {
							case 1: {
										GemmKernel::AccumulatorSet01 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1)
										};
										GemmKernel::maskload_accum_01(load_params, acc);
										GemmKernel::fma_loop_01(ctx);
										GemmKernel::maskstore_accum_01(store_params, acc);
										break;
									}
							case 2: {
										GemmKernel::AccumulatorSet02 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2)
										};
										GemmKernel::maskload_accum_02(load_params, acc);
										GemmKernel::fma_loop_02(ctx);
										GemmKernel::maskstore_accum_02(store_params, acc);
										break;
									}
							case 3: {
										GemmKernel::AccumulatorSet03 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3)
										};
										GemmKernel::maskload_accum_03(load_params, acc);
										GemmKernel::fma_loop_03(ctx);
										GemmKernel::maskstore_accum_03(store_params, acc);
										break;
									}
							case 4: {
										GemmKernel::AccumulatorSet04 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4)
										};
										GemmKernel::maskload_accum_04(load_params, acc);
										GemmKernel::fma_loop_04(ctx);
										GemmKernel::maskstore_accum_04(store_params, acc);
										break;
									}
							case 5: {
										GemmKernel::AccumulatorSet05 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4),
											&ctx.acc(0, 5), &ctx.acc(1, 5)
										};
										GemmKernel::maskload_accum_05(load_params, acc);
										GemmKernel::fma_loop_05(ctx);
										GemmKernel::maskstore_accum_05(store_params, acc);
										break;
									}
						}
					} else {
						GemmKernel::LoadParams load_params = { .C = C, .M = M };
						GemmKernel::StoreParams store_params = load_params;

						switch (nr) {
							case 1: {
										GemmKernel::AccumulatorSet01 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1)
										};
										GemmKernel::load_accum_01(load_params, acc);
										GemmKernel::fma_loop_01(ctx);
										GemmKernel::store_accum_01(store_params, acc);
										break;
									}
							case 2: {
										GemmKernel::AccumulatorSet02 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2)
										};
										GemmKernel::load_accum_02(load_params, acc);
										GemmKernel::fma_loop_02(ctx);
										GemmKernel::store_accum_02(store_params, acc);
										break;
									}
							case 3: {
										GemmKernel::AccumulatorSet03 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3)
										};
										GemmKernel::load_accum_03(load_params, acc);
										GemmKernel::fma_loop_03(ctx);
										GemmKernel::store_accum_03(store_params, acc);
										break;
									}
							case 4: {
										GemmKernel::AccumulatorSet04 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4)
										};
										GemmKernel::load_accum_04(load_params, acc);
										GemmKernel::fma_loop_04(ctx);
										GemmKernel::store_accum_04(store_params, acc);
										break;
									}
							case 5: {
										GemmKernel::AccumulatorSet05 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4),
											&ctx.acc(0, 5), &ctx.acc(1, 5)
										};
										GemmKernel::load_accum_05(load_params, acc);
										GemmKernel::fma_loop_05(ctx);
										GemmKernel::store_accum_05(store_params, acc);
										break;
									}
						}
					}
				}

				static  inline void kernel_16x6_zero_init_accum(float* blockA_packed,
						float* blockB_packed,
						float* C,
						int mr,
						int nr,
						int kc,
						int M)
				{
					FmaContext ctx = {
						.a = blockA_packed,
						.b = blockB_packed,
						.kc = kc
					};

					__m256i mask0 = {};
					__m256i mask1 = {};

					if (mr != 16) {
						build_masks(&mask0, &mask1, mr);
						MaskedStoreParams store_params = {
							.C = C,
							.M = M,
							.mask0 = mask0,
							.mask1 = mask1
						};

						switch (nr) {
							case 1: {
										AccumulatorSet01 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1)
										};
										fma_loop_01(ctx);
										maskstore_accum_01(store_params, acc);
										break;
									}
							case 2: {
										AccumulatorSet02 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2)
										};
										fma_loop_02(ctx);
										maskstore_accum_02(store_params, acc);
										break;
									}
							case 3: {
										AccumulatorSet03 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3)
										};
										fma_loop_03(ctx);
										maskstore_accum_03(store_params, acc);
										break;
									}
							case 4: {
										AccumulatorSet04 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4)
										};
										fma_loop_04(ctx);
										maskstore_accum_04(store_params, acc);
										break;
									}
							case 5: {
										AccumulatorSet05 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4),
											&ctx.acc(0, 5), &ctx.acc(1, 5)
										};
										fma_loop_05(ctx);
										maskstore_accum_05(store_params, acc);
										break;
									}
						}
					} else {
						StoreParams store_params = {
							.C = C,
							.M = M
						};

						switch (nr) {
							case 1: {
										AccumulatorSet01 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1)
										};
										fma_loop_01(ctx);
										store_accum_01(store_params, acc);
										break;
									}
							case 2: {
										AccumulatorSet02 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2)
										};
										fma_loop_02(ctx);
										store_accum_02(store_params, acc);
										break;
									}
							case 3: {
										AccumulatorSet03 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3)
										};
										fma_loop_03(ctx);
										store_accum_03(store_params, acc);
										break;
									}
							case 4: {
										AccumulatorSet04 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4)
										};
										fma_loop_04(ctx);
										store_accum_04(store_params, acc);
										break;
									}
							case 5: {
										AccumulatorSet05 acc = {
											&ctx.acc(0, 0), &ctx.acc(1, 0),
											&ctx.acc(0, 1), &ctx.acc(1, 1),
											&ctx.acc(0, 2), &ctx.acc(1, 2),
											&ctx.acc(0, 3), &ctx.acc(1, 3),
											&ctx.acc(0, 4), &ctx.acc(1, 4),
											&ctx.acc(0, 5), &ctx.acc(1, 5)
										};
										fma_loop_05(ctx);
										store_accum_05(store_params, acc);
										break;
									}
						}
					}
				}


				static inline void matmul_parallel(float* A, float* B, float* C, int M, int N, int ldb) {
					constexpr int MC = GemmKernel<float>::BlockRows;
					constexpr int NC = GemmKernel<float>::BlockCols;
					constexpr int KC = GemmKernel<float>::BlockDepth;
					constexpr int MR = GemmKernel<float>::TileRows;
					constexpr int NR = GemmKernel<float>::TileCols;

					alignas(64) float blockA_packed[MC * KC] = {};
					alignas(64) float blockB_packed[NC * KC] = {};

					for (int j = 0; j < N; j += NC) {
						int nc = std::min(NC, N - j);
						int kc = std::min(KC, ldb);
						GemmKernel<float>::packBlockB(&B[j * ldb], blockB_packed, nc, kc, ldb);

						for (int i = 0; i < M; i += MC) {
							int mc = std::min(MC, M - i);
							GemmKernel<float>::packBlockA(&A[i], blockA_packed, mc, kc, M);

#pragma omp parallel for collapse(2)
							for (int jr = 0; jr < nc; jr += NR) {
								for (int ir = 0; ir < mc; ir += MR) {
									int nr = std::min(NR, nc - jr);
									int mr = std::min(MR, mc - ir);

									GemmKernel::kernel_16x6_zero_init_accum(
											&blockA_packed[ir * kc],
											&blockB_packed[jr * kc],
											&C[(j + jr) * M + (i + ir)],
											mr, nr, kc, M
											);
								}
							}
						}

						for (int p = kc; p < ldb; p += KC) {
							int pkc = std::min(KC, ldb - p);
							GemmKernel<float>::packBlockB(&B[j * ldb + p], blockB_packed, nc, pkc, ldb);

							for (int i = 0; i < M; i += MC) {
								int mc = std::min(MC, M - i);
								GemmKernel<float>::packBlockA(&A[p * M + i], blockA_packed, mc, pkc, M);

#pragma omp parallel for collapse(2)
								for (int jr = 0; jr < nc; jr += NR) {
									for (int ir = 0; ir < mc; ir += MR) {
										int nr = std::min(NR, nc - jr);
										int mr = std::min(MR, mc - ir);

										GemmKernel::kernel_16x6_load_accum(
												&blockA_packed[ir * pkc],
												&blockB_packed[jr * pkc],
												&C[(j + jr) * M + (i + ir)],
												mr, nr, pkc, M
												);
									}
								}
							}
						}
					}
				}

		};
} // namespace tensorium
