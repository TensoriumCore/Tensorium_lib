#pragma once 

#include "../Matrix.hpp"

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
				

				inline Matrix<K> mul_mat2x2(const Matrix<K>& B) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg = typename Simd::reg;

					Matrix<K> C(2, 2);

					reg b_col0 = Simd::loadu(&B.data[0]); 
					reg b_col1 = Simd::loadu(&B.data[2]);

					K b00 = Simd::extract(b_col0, 0);
					K b10 = Simd::extract(b_col0, 1);
					K b01 = Simd::extract(b_col1, 0);
					K b11 = Simd::extract(b_col1, 1);

					C(0,0) = (*this)(0,0)*b00 + (*this)(0,1)*b10;
					C(1,0) = (*this)(1,0)*b00 + (*this)(1,1)*b10;
					C(0,1) = (*this)(0,0)*b01 + (*this)(0,1)*b11;
					C(1,1) = (*this)(1,0)*b01 + (*this)(1,1)*b11;

					return C;
				}


				inline Matrix<K> mul_mat3x3(const Matrix<K>& mat) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg  = typename Simd::reg;

					Matrix<K> result(3, 3);

					reg c0 = Simd::loadu(&mat.data[0]); 
					reg c1 = Simd::loadu(&mat.data[3]); 
					reg c2 = Simd::loadu(&mat.data[6]); 

					K a00 = (*this)(0,0), a01 = (*this)(0,1), a02 = (*this)(0,2);

					reg r0 = Simd::mul(Simd::set1(a00), c0);
					r0 = Simd::fmadd(Simd::set1(a01), c1, r0);
					r0 = Simd::fmadd(Simd::set1(a02), c2, r0);

					Simd::storeu(&result.data[0], r0); 

					K a10 = (*this)(1,0), a11 = (*this)(1,1), a12 = (*this)(1,2);

					reg r1 = Simd::mul(Simd::set1(a10), c0);
					r1 = Simd::fmadd(Simd::set1(a11), c1, r1);
					r1 = Simd::fmadd(Simd::set1(a12), c2, r1);

					Simd::storeu(&result.data[3], r1); 

					K a20 = (*this)(2,0), a21 = (*this)(2,1), a22 = (*this)(2,2);

					reg r2 = Simd::mul(Simd::set1(a20), c0);
					r2 = Simd::fmadd(Simd::set1(a21), c1, r2);
					r2 = Simd::fmadd(Simd::set1(a22), c2, r2);

					Simd::storeu(&result.data[6], r2); 

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
