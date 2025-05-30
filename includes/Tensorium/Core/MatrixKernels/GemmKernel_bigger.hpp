#pragma once

#include "../Matrix.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "../../MathUtils/MathsUtils.hpp"
/*
 * this Gemm kernel is based on Aman Salykov version. Improvment of the OMP schedulding and Block sizes 
 *
 */

namespace tensorium {
	template<typename T>
		class GemmKernelBigger {
			public:
				using Simd = simd::SimdTraits<T, DefaultISA>;
				using reg = typename Simd::reg;
				static constexpr int SimdWidth = Simd::width;
				static constexpr int TileRows = SimdWidth * 2;
				static constexpr int TileCols = 6; 
				static constexpr int NThreads = 16;

				static constexpr int BlockDepth = 256;   
				static constexpr int BlockRows = 384;   
				static constexpr int BlockCols = 512;   


				static inline int8_t mask[32]
					__attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
						0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

				inline void fma_loop_00(T* blockA_packed,
						T* blockB_packed,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* a0_packFloat8,
						reg* a1_packFloat8,
						reg* b_packFloat8,
						int kc) {

					for (int p = 0; p < kc; p++) {
						*a0_packFloat8 = Simd::loadu(blockA_packed);
						*a1_packFloat8 = Simd::loadu(blockA_packed + 8);

						*b_packFloat8 = Simd::broadcast(blockB_packed);
						*C_accum_00 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_00);
						*C_accum_01 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_01);

						blockA_packed += 16;
						blockB_packed += 6;
					}
				}

				inline void fma_loop_01(T* blockA_packed,
						T* blockB_packed,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* a0_packFloat8,
						reg* a1_packFloat8,
						reg* b_packFloat8,
						int kc) {

					for (int p = 0; p < kc; p++) {
						*a0_packFloat8 = Simd::loadu(blockA_packed);
						*a1_packFloat8 = Simd::loadu(blockA_packed + 8);

						*b_packFloat8 = Simd::broadcast(blockB_packed);
						*C_accum_00 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_00);
						*C_accum_01 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_01);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 1);
						*C_accum_10 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_10);
						*C_accum_11 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_11);

						blockA_packed += 16;
						blockB_packed += 6;
					}
				}

				inline void fma_loop_02(T* blockA_packed,
						T* blockB_packed,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* a0_packFloat8,
						reg* a1_packFloat8,
						reg* b_packFloat8,
						int kc) {

					for (int p = 0; p < kc; p++) {
						*a0_packFloat8 = Simd::loadu(blockA_packed);
						*a1_packFloat8 = Simd::loadu(blockA_packed + 8);

						*b_packFloat8 = Simd::broadcast(blockB_packed);
						*C_accum_00 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_00);
						*C_accum_01 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_01);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 1);
						*C_accum_10 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_10);
						*C_accum_11 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_11);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 2);
						*C_accum_20 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_20);
						*C_accum_21 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_21);

						blockA_packed += 16;
						blockB_packed += 6;
					}
				}

				inline void fma_loop_03(T* blockA_packed,
						T* blockB_packed,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* a0_packFloat8,
						reg* a1_packFloat8,
						reg* b_packFloat8,
						int kc) {

					for (int p = 0; p < kc; p++) {
						*a0_packFloat8 = Simd::loadu(blockA_packed);
						*a1_packFloat8 = Simd::loadu(blockA_packed + 8);

						*b_packFloat8 = Simd::broadcast(blockB_packed);
						*C_accum_00 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_00);
						*C_accum_01 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_01);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 1);
						*C_accum_10 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_10);
						*C_accum_11 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_11);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 2);
						*C_accum_20 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_20);
						*C_accum_21 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_21);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 3);
						*C_accum_30 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_30);
						*C_accum_31 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_31);

						blockA_packed += 16;
						blockB_packed += 6;
					}
				}

				inline void fma_loop_04(T* blockA_packed,
						T* blockB_packed,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						reg* a0_packFloat8,
						reg* a1_packFloat8,
						reg* b_packFloat8,
						int kc) {

					for (int p = 0; p < kc; p++) {
						*a0_packFloat8 = Simd::loadu(blockA_packed);
						*a1_packFloat8 = Simd::loadu(blockA_packed + 8);

						*b_packFloat8 = Simd::broadcast(blockB_packed);
						*C_accum_00 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_00);
						*C_accum_01 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_01);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 1);
						*C_accum_10 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_10);
						*C_accum_11 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_11);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 2);
						*C_accum_20 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_20);
						*C_accum_21 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_21);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 3);
						*C_accum_30 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_30);
						*C_accum_31 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_31);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 4);
						*C_accum_40 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_40);
						*C_accum_41 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_41);

						blockA_packed += 16;
						blockB_packed += 6;
					}
				}

				inline void fma_loop_05(T* blockA_packed,
						T* blockB_packed,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						reg* C_accum_50,
						reg* C_accum_51,
						reg* a0_packFloat8,
						reg* a1_packFloat8,
						reg* b_packFloat8,
						int kc) {

					for (int p = 0; p < kc; p++) {
						*a0_packFloat8 = Simd::loadu(blockA_packed);
						*a1_packFloat8 = Simd::loadu(blockA_packed + 8);

						*b_packFloat8 = Simd::broadcast(blockB_packed);
						*C_accum_00 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_00);
						*C_accum_01 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_01);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 1);
						*C_accum_10 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_10);
						*C_accum_11 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_11);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 2);
						*C_accum_20 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_20);
						*C_accum_21 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_21);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 3);
						*C_accum_30 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_30);
						*C_accum_31 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_31);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 4);
						*C_accum_40 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_40);
						*C_accum_41 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_41);

						*b_packFloat8 = Simd::broadcast(blockB_packed + 5);
						*C_accum_50 = Simd::fmadd(*a0_packFloat8, *b_packFloat8, *C_accum_50);
						*C_accum_51 = Simd::fmadd(*a1_packFloat8, *b_packFloat8, *C_accum_51);

						blockA_packed += 16;
						blockB_packed += 6;
					}
				}


				inline static void build_masks(__m256i* packed_mask_0, __m256i* packed_mask_1, int mr) {
#if defined(__AVX512F__)
					__m128i m0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&mask[32 - mr]));
					__m128i m1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&mask[32 - mr + 16]));

					__m512i p0 = _mm512_cvtepi8_epi32(m0);
					__m512i p1 = _mm512_cvtepi8_epi32(m1);

					*packed_mask_0 = _mm512_castsi512_si256(p0);
					*packed_mask_1 = _mm512_castsi512_si256(p1);

#elif defined(__AVX2__)
					__m128i m0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&mask[16 - mr]));
					__m128i m1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&mask[16 - mr + 8]));

					*packed_mask_0 = _mm256_cvtepi8_epi32(m0);
					*packed_mask_1 = _mm256_cvtepi8_epi32(m1);
#else
#error "AVX2 or AVX-512 required"
#endif
				}


				inline void maskload_accum_00(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					*C_accum_00 = Simd::maskload(C, packed_mask_0);
					*C_accum_01 = Simd::maskload(&C[8], packed_mask_1);
				}

				inline void maskload_accum_01(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					*C_accum_00 = Simd::maskload(C, packed_mask_0);
					*C_accum_01 = Simd::maskload(&C[8], packed_mask_1);
					*C_accum_10 = Simd::maskload(&C[M], packed_mask_0);
					*C_accum_11 = Simd::maskload(&C[M + 8], packed_mask_1);
				}

				inline void maskload_accum_02(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					*C_accum_00 = Simd::maskload(C, packed_mask_0);
					*C_accum_01 = Simd::maskload(&C[8], packed_mask_1);
					*C_accum_10 = Simd::maskload(&C[M], packed_mask_0);
					*C_accum_11 = Simd::maskload(&C[M + 8], packed_mask_1);
					*C_accum_20 = Simd::maskload(&C[2 * M], packed_mask_0);
					*C_accum_21 = Simd::maskload(&C[2 * M + 8], packed_mask_1);
				}

				inline void maskload_accum_03(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					*C_accum_00 = Simd::maskload(C, packed_mask_0);
					*C_accum_01 = Simd::maskload(&C[8], packed_mask_1);
					*C_accum_10 = Simd::maskload(&C[M], packed_mask_0);
					*C_accum_11 = Simd::maskload(&C[M + 8], packed_mask_1);
					*C_accum_20 = Simd::maskload(&C[2 * M], packed_mask_0);
					*C_accum_21 = Simd::maskload(&C[2 * M + 8], packed_mask_1);
					*C_accum_30 = Simd::maskload(&C[3 * M], packed_mask_0);
					*C_accum_31 = Simd::maskload(&C[3 * M + 8], packed_mask_1);
				}

				inline void maskload_accum_04(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					*C_accum_00 = Simd::maskload(C, packed_mask_0);
					*C_accum_01 = Simd::maskload(&C[8], packed_mask_1);
					*C_accum_10 = Simd::maskload(&C[M], packed_mask_0);
					*C_accum_11 = Simd::maskload(&C[M + 8], packed_mask_1);
					*C_accum_20 = Simd::maskload(&C[2 * M], packed_mask_0);
					*C_accum_21 = Simd::maskload(&C[2 * M + 8], packed_mask_1);
					*C_accum_30 = Simd::maskload(&C[3 * M], packed_mask_0);
					*C_accum_31 = Simd::maskload(&C[3 * M + 8], packed_mask_1);
					*C_accum_40 = Simd::maskload(&C[4 * M], packed_mask_0);
					*C_accum_41 = Simd::maskload(&C[4 * M + 8], packed_mask_1);
				}

				inline void maskload_accum_05(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						reg* C_accum_50,
						reg* C_accum_51,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					*C_accum_00 = Simd::maskload(C, packed_mask_0);
					*C_accum_01 = Simd::maskload(&C[8], packed_mask_1);
					*C_accum_10 = Simd::maskload(&C[M], packed_mask_0);
					*C_accum_11 = Simd::maskload(&C[M + 8], packed_mask_1);
					*C_accum_20 = Simd::maskload(&C[2 * M], packed_mask_0);
					*C_accum_21 = Simd::maskload(&C[2 * M + 8], packed_mask_1);
					*C_accum_30 = Simd::maskload(&C[3 * M], packed_mask_0);
					*C_accum_31 = Simd::maskload(&C[3 * M + 8], packed_mask_1);
					*C_accum_40 = Simd::maskload(&C[4 * M], packed_mask_0);
					*C_accum_41 = Simd::maskload(&C[4 * M + 8], packed_mask_1);
					*C_accum_50 = Simd::maskload(&C[5 * M], packed_mask_0);
					*C_accum_51 = Simd::maskload(&C[5 * M + 8], packed_mask_1);
				}

				inline void load_accum_00(T* C, reg* C_accum_00, reg* C_accum_01, int M) {
					*C_accum_00 = Simd::loadu(C);
					*C_accum_01 = Simd::loadu(&C[8]);
				}

				inline void load_accum_01(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						int M) {
					*C_accum_00 = Simd::loadu(C);
					*C_accum_01 = Simd::loadu(&C[8]);
					*C_accum_10 = Simd::loadu(&C[M]);
					*C_accum_11 = Simd::loadu(&C[M + 8]);
				}

				inline void load_accum_02(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						int M) {
					*C_accum_00 = Simd::loadu(C);
					*C_accum_01 = Simd::loadu(&C[8]);
					*C_accum_10 = Simd::loadu(&C[M]);
					*C_accum_11 = Simd::loadu(&C[M + 8]);
					*C_accum_20 = Simd::loadu(&C[2 * M]);
					*C_accum_21 = Simd::loadu(&C[2 * M + 8]);
				}

				inline void load_accum_03(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						int M) {
					*C_accum_00 = Simd::loadu(C);
					*C_accum_01 = Simd::loadu(&C[8]);
					*C_accum_10 = Simd::loadu(&C[M]);
					*C_accum_11 = Simd::loadu(&C[M + 8]);
					*C_accum_20 = Simd::loadu(&C[2 * M]);
					*C_accum_21 = Simd::loadu(&C[2 * M + 8]);
					*C_accum_30 = Simd::loadu(&C[3 * M]);
					*C_accum_31 = Simd::loadu(&C[3 * M + 8]);
				}

				inline void load_accum_04(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						int M) {
					*C_accum_00 = Simd::loadu(C);
					*C_accum_01 = Simd::loadu(&C[8]);
					*C_accum_10 = Simd::loadu(&C[M]);
					*C_accum_11 = Simd::loadu(&C[M + 8]);
					*C_accum_20 = Simd::loadu(&C[2 * M]);
					*C_accum_21 = Simd::loadu(&C[2 * M + 8]);
					*C_accum_30 = Simd::loadu(&C[3 * M]);
					*C_accum_31 = Simd::loadu(&C[3 * M + 8]);
					*C_accum_40 = Simd::loadu(&C[4 * M]);
					*C_accum_41 = Simd::loadu(&C[4 * M + 8]);
				}

				inline void load_accum_05(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						reg* C_accum_50,
						reg* C_accum_51,
						int M) {
					*C_accum_00 = Simd::loadu(C);
					*C_accum_01 = Simd::loadu(&C[8]);
					*C_accum_10 = Simd::loadu(&C[M]);
					*C_accum_11 = Simd::loadu(&C[M + 8]);
					*C_accum_20 = Simd::loadu(&C[2 * M]);
					*C_accum_21 = Simd::loadu(&C[2 * M + 8]);
					*C_accum_30 = Simd::loadu(&C[3 * M]);
					*C_accum_31 = Simd::loadu(&C[3 * M + 8]);
					*C_accum_40 = Simd::loadu(&C[4 * M]);
					*C_accum_41 = Simd::loadu(&C[4 * M + 8]);
					*C_accum_50 = Simd::loadu(&C[5 * M]);
					*C_accum_51 = Simd::loadu(&C[5 * M + 8]);
				}

				inline void store_accum_00(T* C, reg* C_accum_00, reg* C_accum_01, int M) {
					Simd::storeu(C, *C_accum_00);
					Simd::storeu(&C[8], *C_accum_01);
				}

				inline void store_accum_01(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						int M) {
					Simd::storeu(C, *C_accum_00);
					Simd::storeu(&C[8], *C_accum_01);
					Simd::storeu(&C[M], *C_accum_10);
					Simd::storeu(&C[M + 8], *C_accum_11);
				}

				inline void store_accum_02(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						int M) {
					Simd::storeu(C, *C_accum_00);
					Simd::storeu(&C[8], *C_accum_01);
					Simd::storeu(&C[M], *C_accum_10);
					Simd::storeu(&C[M + 8], *C_accum_11);
					Simd::storeu(&C[2 * M], *C_accum_20);
					Simd::storeu(&C[2 * M + 8], *C_accum_21);
				}

				inline void store_accum_03(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						int M) {
					Simd::storeu(C, *C_accum_00);
					Simd::storeu(&C[8], *C_accum_01);
					Simd::storeu(&C[M], *C_accum_10);
					Simd::storeu(&C[M + 8], *C_accum_11);
					Simd::storeu(&C[2 * M], *C_accum_20);
					Simd::storeu(&C[2 * M + 8], *C_accum_21);
					Simd::storeu(&C[3 * M], *C_accum_30);
					Simd::storeu(&C[3 * M + 8], *C_accum_31);
				}

				inline void store_accum_04(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						int M) {
					Simd::storeu(C, *C_accum_00);
					Simd::storeu(&C[8], *C_accum_01);
					Simd::storeu(&C[M], *C_accum_10);
					Simd::storeu(&C[M + 8], *C_accum_11);
					Simd::storeu(&C[2 * M], *C_accum_20);
					Simd::storeu(&C[2 * M + 8], *C_accum_21);
					Simd::storeu(&C[3 * M], *C_accum_30);
					Simd::storeu(&C[3 * M + 8], *C_accum_31);
					Simd::storeu(&C[4 * M], *C_accum_40);
					Simd::storeu(&C[4 * M + 8], *C_accum_41);
				}

				inline void store_accum_05(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						reg* C_accum_50,
						reg* C_accum_51,
						int M) {
					Simd::storeu(C, *C_accum_00);
					Simd::storeu(&C[8], *C_accum_01);
					Simd::storeu(&C[M], *C_accum_10);
					Simd::storeu(&C[M + 8], *C_accum_11);
					Simd::storeu(&C[2 * M], *C_accum_20);
					Simd::storeu(&C[2 * M + 8], *C_accum_21);
					Simd::storeu(&C[3 * M], *C_accum_30);
					Simd::storeu(&C[3 * M + 8], *C_accum_31);
					Simd::storeu(&C[4 * M], *C_accum_40);
					Simd::storeu(&C[4 * M + 8], *C_accum_41);
					Simd::storeu(&C[5 * M], *C_accum_50);
					Simd::storeu(&C[5 * M + 8], *C_accum_51);
				}

				inline void maskstore_accum_00(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					Simd::maskstore(C, packed_mask_0, *C_accum_00);
					Simd::maskstore(&C[8], packed_mask_1, *C_accum_01);
				}

				inline void maskstore_accum_01(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					Simd::maskstore(C, packed_mask_0, *C_accum_00);
					Simd::maskstore(&C[8], packed_mask_1, *C_accum_01);
					Simd::maskstore(&C[M], packed_mask_0, *C_accum_10);
					Simd::maskstore(&C[M + 8], packed_mask_1, *C_accum_11);
				}

				inline void maskstore_accum_02(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					Simd::maskstore(C, packed_mask_0, *C_accum_00);
					Simd::maskstore(&C[8], packed_mask_1, *C_accum_01);
					Simd::maskstore(&C[M], packed_mask_0, *C_accum_10);
					Simd::maskstore(&C[M + 8], packed_mask_1, *C_accum_11);
					Simd::maskstore(&C[2 * M], packed_mask_0, *C_accum_20);
					Simd::maskstore(&C[2 * M + 8], packed_mask_1, *C_accum_21);
				}

				inline void maskstore_accum_03(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					Simd::maskstore(C, packed_mask_0, *C_accum_00);
					Simd::maskstore(&C[8], packed_mask_1, *C_accum_01);
					Simd::maskstore(&C[M], packed_mask_0, *C_accum_10);
					Simd::maskstore(&C[M + 8], packed_mask_1, *C_accum_11);
					Simd::maskstore(&C[2 * M], packed_mask_0, *C_accum_20);
					Simd::maskstore(&C[2 * M + 8], packed_mask_1, *C_accum_21);
					Simd::maskstore(&C[3 * M], packed_mask_0, *C_accum_30);
					Simd::maskstore(&C[3 * M + 8], packed_mask_1, *C_accum_31);
				}

				inline void maskstore_accum_04(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					Simd::maskstore(C, packed_mask_0, *C_accum_00);
					Simd::maskstore(&C[8], packed_mask_1, *C_accum_01);
					Simd::maskstore(&C[M], packed_mask_0, *C_accum_10);
					Simd::maskstore(&C[M + 8], packed_mask_1, *C_accum_11);
					Simd::maskstore(&C[2 * M], packed_mask_0, *C_accum_20);
					Simd::maskstore(&C[2 * M + 8], packed_mask_1, *C_accum_21);
					Simd::maskstore(&C[3 * M], packed_mask_0, *C_accum_30);
					Simd::maskstore(&C[3 * M + 8], packed_mask_1, *C_accum_31);
					Simd::maskstore(&C[4 * M], packed_mask_0, *C_accum_40);
					Simd::maskstore(&C[4 * M + 8], packed_mask_1, *C_accum_41);
				}

				inline void maskstore_accum_05(T* C,
						reg* C_accum_00,
						reg* C_accum_01,
						reg* C_accum_10,
						reg* C_accum_11,
						reg* C_accum_20,
						reg* C_accum_21,
						reg* C_accum_30,
						reg* C_accum_31,
						reg* C_accum_40,
						reg* C_accum_41,
						reg* C_accum_50,
						reg* C_accum_51,
						__m256i packed_mask_0,
						__m256i packed_mask_1,
						int M) {
					Simd::maskstore(C, packed_mask_0, *C_accum_00);
					Simd::maskstore(&C[8], packed_mask_1, *C_accum_01);
					Simd::maskstore(&C[M], packed_mask_0, *C_accum_10);
					Simd::maskstore(&C[M + 8], packed_mask_1, *C_accum_11);
					Simd::maskstore(&C[2 * M], packed_mask_0, *C_accum_20);
					Simd::maskstore(&C[2 * M + 8], packed_mask_1, *C_accum_21);
					Simd::maskstore(&C[3 * M], packed_mask_0, *C_accum_30);
					Simd::maskstore(&C[3 * M + 8], packed_mask_1, *C_accum_31);
					Simd::maskstore(&C[4 * M], packed_mask_0, *C_accum_40);
					Simd::maskstore(&C[4 * M + 8], packed_mask_1, *C_accum_41);
					Simd::maskstore(&C[5 * M], packed_mask_0, *C_accum_50);
					Simd::maskstore(&C[5 * M + 8], packed_mask_1, *C_accum_51);
				}

				void kernel_16x6_load_accum(T* blockA_packed,
						T* blockB_packed,
						T* C,
						int mr,
						int nr,
						int kc,
						int M) {
					reg C_accum_00 = {};
					reg C_accum_01 = {};
					reg C_accum_10 = {};
					reg C_accum_11 = {};
					reg C_accum_20 = {};
					reg C_accum_21 = {};
					reg C_accum_30 = {};
					reg C_accum_31 = {};
					reg C_accum_40 = {};
					reg C_accum_41 = {};
					reg C_accum_50 = {};
					reg C_accum_51 = {};

					reg b_packFloat8 = {};
					reg a0_packFloat8 = {};
					reg a1_packFloat8 = {};
					__m256i packed_mask_0 = {};
					__m256i packed_mask_1 = {};

					if (mr != 16) {
						build_masks(&packed_mask_0, &packed_mask_1, mr);
						switch (nr) {
							case 1 :
								maskload_accum_00(C, &C_accum_00, &C_accum_01, packed_mask_0, packed_mask_1, M);
								fma_loop_00(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_00(C, &C_accum_00, &C_accum_01, packed_mask_0, packed_mask_1, M);
								break;
							case 2 :
								maskload_accum_01(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										packed_mask_0,
										packed_mask_1,
										M);
								fma_loop_01(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_01(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 3 :
								maskload_accum_02(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										packed_mask_0,
										packed_mask_1,
										M);
								fma_loop_02(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_02(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 4 :
								maskload_accum_03(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										packed_mask_0,
										packed_mask_1,
										M);
								fma_loop_03(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_03(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 5 :
								maskload_accum_04(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										packed_mask_0,
										packed_mask_1,
										M);
								fma_loop_04(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_04(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 6 :
								maskload_accum_05(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										packed_mask_0,
										packed_mask_1,
										M);
								fma_loop_05(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_05(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
						}
					} else {
						switch (nr) {
							case 1 :
								load_accum_00(C, &C_accum_00, &C_accum_01, M);
								fma_loop_00(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_00(C, &C_accum_00, &C_accum_01, M);
								break;
							case 2 :
								load_accum_01(C, &C_accum_00, &C_accum_01, &C_accum_10, &C_accum_11, M);
								fma_loop_01(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_01(C, &C_accum_00, &C_accum_01, &C_accum_10, &C_accum_11, M);
								break;
							case 3 :
								load_accum_02(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										M);
								fma_loop_02(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_02(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										M);
								break;
							case 4 :
								load_accum_03(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										M);
								fma_loop_03(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_03(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										M);
								break;
							case 5 :
								load_accum_04(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										M);
								fma_loop_04(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_04(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										M);

								break;
							case 6 :
								load_accum_05(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										M);
								fma_loop_05(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_05(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										M);
								break;
						}
					}
				}

				void kernel_16x6_zero_init_accum(T* blockA_packed,
						T* blockB_packed,
						T* C,
						int mr,
						int nr,
						int kc,
						int M) {
					reg C_accum_00 = {};
					reg C_accum_01 = {};
					reg C_accum_10 = {};
					reg C_accum_11 = {};
					reg C_accum_20 = {};
					reg C_accum_21 = {};
					reg C_accum_30 = {};
					reg C_accum_31 = {};
					reg C_accum_40 = {};
					reg C_accum_41 = {};
					reg C_accum_50 = {};
					reg C_accum_51 = {};

					reg b_packFloat8 = {};
					reg a0_packFloat8 = {};
					reg a1_packFloat8 = {};
					__m256i packed_mask_0 = {};
					__m256i packed_mask_1 = {};

					if (mr != 16) {
						build_masks(&packed_mask_0, &packed_mask_1, mr);
						switch (nr) {
							case 1 :
								fma_loop_00(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_00(C, &C_accum_00, &C_accum_01, packed_mask_0, packed_mask_1, M);
								break;
							case 2 :
								fma_loop_01(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_01(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 3 :
								fma_loop_02(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_02(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 4 :
								fma_loop_03(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_03(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 5 :
								fma_loop_04(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_04(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
							case 6 :
								fma_loop_05(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								maskstore_accum_05(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										packed_mask_0,
										packed_mask_1,
										M);
								break;
						}
					} else {
						switch (nr) {
							case 1 :
								fma_loop_00(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_00(C, &C_accum_00, &C_accum_01, M);
								break;
							case 2 :
								fma_loop_01(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_01(C, &C_accum_00, &C_accum_01, &C_accum_10, &C_accum_11, M);
								break;
							case 3 :
								fma_loop_02(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_02(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										M);
								break;
							case 4 :
								fma_loop_03(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_03(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										M);
								break;
							case 5 :
								fma_loop_04(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_04(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										M);

								break;
							case 6 :
								fma_loop_05(blockA_packed,
										blockB_packed,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										&a0_packFloat8,
										&a1_packFloat8,
										&b_packFloat8,
										kc);
								store_accum_05(C,
										&C_accum_00,
										&C_accum_01,
										&C_accum_10,
										&C_accum_11,
										&C_accum_20,
										&C_accum_21,
										&C_accum_30,
										&C_accum_31,
										&C_accum_40,
										&C_accum_41,
										&C_accum_50,
										&C_accum_51,
										M);
								break;
						}
					}
				}




#ifndef NTHREADS
#define NTHREADS 8
#endif

#define MC (16 * (40 / NTHREADS) * NTHREADS)
#define NC (6 * (800 / NTHREADS) * NTHREADS)
#define KC 500

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE auto
#endif
#define _min(x, y) ((x) < (y) ? (x) : (y))
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")


				static T blockA_packed[MC * KC] __attribute__((aligned(64)));
				static T blockB_packed[NC * KC] __attribute__((aligned(64)));

				void pack_panelB(T* B, T* blockB_packed, int nr, int kc, int K) {
					for (int p = 0; p < kc; p++) {
						for (int j = 0; j < nr; j++) {
							*blockB_packed++ = B[j * K + p];
						}
						for (int j = nr; j < 6; j++) {
							*blockB_packed++ = 0;
						}
					}
				}

				void pack_blockB(T* B, T* blockB_packed, int nc, int kc, int K) {
#pragma omp for schedule(dynamic)
					for (int j = 0; j < nc; j += 6) {
						int nr = _min(6, nc - j);
						pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
					}
				}

				void pack_panelA(T* A, T* blockA_packed, int mr, int kc, int M) {
					for (int p = 0; p < kc; p++) {
						for (int i = 0; i < mr; i++) {
							*blockA_packed++ = A[p * M + i];
						}
						for (int i = mr; i < 16; i++) {
							*blockA_packed++ = 0;
						}
					}
				}

				void pack_blockA(T* A, T* blockA_packed, int mc, int kc, int M) {
					PRAGMA_OMP_PARALLEL_FOR
						for (int i = 0; i < mc; i += 16) {
							int mr = _min(16, mc - i);
							pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
						}
				}

				void matmul(T* A, T* B, T* C, int M, int N, int K) {
					for (int j = 0; j < N; j += NC) {
						int nc = _min(NC, N - j);
						int kc = _min(KC, K);

						pack_blockB(&B[j * K], blockB_packed, nc, kc, K);

						for (int i = 0; i < M; i += MC) {
							int mc = _min(MC, M - i);

							pack_blockA(&A[i], blockA_packed, mc, kc, M);

							PRAGMA_OMP_PARALLEL_FOR
								for (int jr = 0; jr < nc; jr += 6) {
									int nr = _min(6, nc - jr);
									for (int ir = 0; ir < mc; ir += 16) {
										int mr = _min(16, mc - ir);
										kernel_16x6_zero_init_accum(
												&blockA_packed[ir * kc],
												&blockB_packed[jr * kc],
												&C[(j + jr) * M + (i + ir)],
												mr, nr, kc, M);
									}
								}
						}
						for (int p = kc; p < K; p += KC) {
							int cur_kc = _min(KC, K - p);
							pack_blockB(&B[j * K + p], blockB_packed, nc, cur_kc, K);

							for (int i = 0; i < M; i += MC) {
								int mc = _min(MC, M - i);

								pack_blockA(&A[i + p * M], blockA_packed, mc, cur_kc, M);

								PRAGMA_OMP_PARALLEL_FOR
									for (int jr = 0; jr < nc; jr += 6) {
										int nr = _min(6, nc - jr);
										for (int ir = 0; ir < mc; ir += 16) {
											int mr = _min(16, mc - ir);
											kernel_16x6_load_accum(
													&blockA_packed[ir * cur_kc],
													&blockB_packed[jr * cur_kc],
													&C[(j + jr) * M + (i + ir)],
													mr, nr, cur_kc, M);
										}
									}
							}
						}
					}
				}
		};
} 


namespace tensorium {
	template<typename T>
		T GemmKernelBigger<T>::blockA_packed[MC * KC] __attribute__((aligned(64)));

	template<typename T>
		T GemmKernelBigger<T>::blockB_packed[NC * KC] __attribute__((aligned(64)));
}

