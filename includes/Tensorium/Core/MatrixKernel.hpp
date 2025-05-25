#pragma once 

#include "Matrix.hpp"

namespace tensorium { 
	template <typename K>
		class MatrixKernel : public Matrix<K> {
			public :
				using Matrix<K>::rows;
				using Matrix<K>::cols;
				using Matrix<K>::data;
				using Matrix<K>::operator();

				using Simd = simd::SimdTraits<K, DefaultISA>;
				using reg  = typename Simd::reg;

				MatrixKernel(size_t r, size_t c) : Matrix<K>(r, c) {}
				MatrixKernel(const Matrix<K>& m) : Matrix<K>(m) {}
				
				inline Matrix<K> mul_mat2x2(const Matrix<K>& mat) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg = typename Simd::reg;

					Matrix<K> result(2, 2);

					reg a_row0 = Simd::loadu(&data[0]);
					reg a_row1 = Simd::loadu(&data[2]); 

					reg b_row0 = Simd::loadu(&mat.data[0]);
					reg b_row1 = Simd::loadu(&mat.data[2]);

					reg a00 = Simd::set1(data[0]); 
					reg a01 = Simd::set1(data[1]); 
					reg acc0 = Simd::mul(a00, b_row0); 
					acc0 = Simd::fmadd(a01, b_row1, acc0); 
					Simd::storeu(&result.data[0], acc0);

					reg a10 = Simd::set1(data[2]); 
					reg a11 = Simd::set1(data[3]);
					reg acc1 = Simd::mul(a10, b_row0);
					acc1 = Simd::fmadd(a11, b_row1, acc1);
					Simd::storeu(&result.data[2], acc1);

					return result;
				}

				inline Matrix<K> mul_mat3x3(const Matrix<K>& mat) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg  = typename Simd::reg;

					Matrix<K> result(3, 3);

					reg row0 = Simd::loadu(&data[0]);
					reg row1 = Simd::loadu(&data[3]);
					reg row2 = Simd::loadu(&data[6]);

					reg c0 = Simd::loadu(&mat.data[0]);
					reg c1 = Simd::loadu(&mat.data[3]);
					reg c2 = Simd::loadu(&mat.data[6]);

					K x0 = data[0];
					K y0 = data[1];
					K z0 = data[2];

					reg sx0 = Simd::set1(x0);
					reg sy0 = Simd::set1(y0);
					reg sz0 = Simd::set1(z0);

					reg acc0 = Simd::mul(sx0, c0);
					acc0 = Simd::fmadd(sy0, c1, acc0);
					acc0 = Simd::fmadd(sz0, c2, acc0);

					Simd::storeu(&result.data[0], acc0);

					K x1 = data[3];
					K y1 = data[4];
					K z1 = data[5];

					reg sx1 = Simd::set1(x1);
					reg sy1 = Simd::set1(y1);
					reg sz1 = Simd::set1(z1);

					reg acc1 = Simd::mul(sx1, c0);
					acc1 = Simd::fmadd(sy1, c1, acc1);
					acc1 = Simd::fmadd(sz1, c2, acc1);

					Simd::storeu(&result.data[3], acc1);

					K x2 = data[6];
					K y2 = data[7];
					K z2 = data[8];

					reg sx2 = Simd::set1(x2);
					reg sy2 = Simd::set1(y2);
					reg sz2 = Simd::set1(z2);

					reg acc2 = Simd::mul(sx2, c0);
					acc2 = Simd::fmadd(sy2, c1, acc2);
					acc2 = Simd::fmadd(sz2, c2, acc2);

					Simd::storeu(&result.data[6], acc2);

					return result;
				}


				inline Matrix<K> mul_mat4x4(const Matrix<K>& B) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg  = typename Simd::reg;

					Matrix<K> result(4, 4);

					reg brow0 = Simd::loadu(&B.data[0*4 + 0]); 
					reg brow1 = Simd::loadu(&B.data[1*4 + 0]); 
					reg brow2 = Simd::loadu(&B.data[2*4 + 0]); 
					reg brow3 = Simd::loadu(&B.data[3*4 + 0]); 

					{
						const K* a = &data[0*4];
						reg a0 = Simd::set1(a[0]);
						reg a1 = Simd::set1(a[1]);
						reg a2 = Simd::set1(a[2]);
						reg a3 = Simd::set1(a[3]);

						reg acc0 = Simd::mul(a0, brow0);
						acc0     = Simd::fmadd(a1, brow1, acc0);
						acc0     = Simd::fmadd(a2, brow2, acc0);
						acc0     = Simd::fmadd(a3, brow3, acc0);

						Simd::storeu(&result.data[0*4 + 0], acc0);
					}

					{
						const K* a = &data[1*4];
						reg a0 = Simd::set1(a[0]);
						reg a1 = Simd::set1(a[1]);
						reg a2 = Simd::set1(a[2]);
						reg a3 = Simd::set1(a[3]);

						reg acc1 = Simd::mul(a0, brow0);
						acc1     = Simd::fmadd(a1, brow1, acc1);
						acc1     = Simd::fmadd(a2, brow2, acc1);
						acc1     = Simd::fmadd(a3, brow3, acc1);

						Simd::storeu(&result.data[1*4 + 0], acc1);
					}

					{
						const K* a = &data[2*4];
						reg a0 = Simd::set1(a[0]);
						reg a1 = Simd::set1(a[1]);
						reg a2 = Simd::set1(a[2]);
						reg a3 = Simd::set1(a[3]);

						reg acc2 = Simd::mul(a0, brow0);
						acc2     = Simd::fmadd(a1, brow1, acc2);
						acc2     = Simd::fmadd(a2, brow2, acc2);
						acc2     = Simd::fmadd(a3, brow3, acc2);

						Simd::storeu(&result.data[2*4 + 0], acc2);
					}

					{
						const K* a = &data[3*4];
						reg a0 = Simd::set1(a[0]);
						reg a1 = Simd::set1(a[1]);
						reg a2 = Simd::set1(a[2]);
						reg a3 = Simd::set1(a[3]);

						reg acc3 = Simd::mul(a0, brow0);
						acc3     = Simd::fmadd(a1, brow1, acc3);
						acc3     = Simd::fmadd(a2, brow2, acc3);
						acc3     = Simd::fmadd(a3, brow3, acc3);

						Simd::storeu(&result.data[3*4 + 0], acc3);
					}

					return result;
				}

				inline Matrix<K> mul_mat8x8(const Matrix<K>& B) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg  = typename Simd::reg;

					Matrix<K> result(8, 8);

					reg col[8];
					for (int j = 0; j < 8; ++j) 
						col[j] = Simd::loadu(&B.data[j * 8]);

					for (int i = 0; i < 8; ++i) {
						const K* a = &data[i * 8];
						reg a0 = Simd::set1(a[0]);
						reg a1 = Simd::set1(a[1]);
						reg a2 = Simd::set1(a[2]);
						reg a3 = Simd::set1(a[3]);
						reg a4 = Simd::set1(a[4]);
						reg a5 = Simd::set1(a[5]);
						reg a6 = Simd::set1(a[6]);
						reg a7 = Simd::set1(a[7]);

						reg acc = Simd::mul(a0, col[0]);
						acc = Simd::fmadd(a1, col[1], acc);
						acc = Simd::fmadd(a2, col[2], acc);
						acc = Simd::fmadd(a3, col[3], acc);
						acc = Simd::fmadd(a4, col[4], acc);
						acc = Simd::fmadd(a5, col[5], acc);
						acc = Simd::fmadd(a6, col[6], acc);
						acc = Simd::fmadd(a7, col[7], acc);

						Simd::storeu(&result.data[i * 8], acc);
					}
					return result;
				}


				inline Matrix<K> mul_mat16x16(const Matrix<K>& B) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg  = typename Simd::reg;

					Matrix<K> result(16, 16);

					reg row_lo[16], row_hi[16];
					for (int k = 0; k < 16; ++k) {
						row_lo[k] = Simd::loadu(&B.data[k*16 + 0]);
						row_hi[k] = Simd::loadu(&B.data[k*16 + 8]);
					}

					for (int i = 0; i < 16; ++i) {
						const K* a = &data[i*16];

						reg acc_lo = Simd::mul( Simd::set1(a[0]), row_lo[0] );
						reg acc_hi = Simd::mul( Simd::set1(a[0]), row_hi[0] );

						for (int k = 1; k < 16; ++k) {
							reg ak = Simd::set1(a[k]);
							acc_lo = Simd::fmadd(ak, row_lo[k], acc_lo);
							acc_hi = Simd::fmadd(ak, row_hi[k], acc_hi);
						}

						Simd::storeu(&result.data[i*16 +   0], acc_lo);
						Simd::storeu(&result.data[i*16 +   8], acc_hi);
					}

					return result;
				}

		};
}
