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

					reg b0 = Simd::set(B(0, 0), B(1, 0), B(2, 0), B(3, 0));
					reg b1 = Simd::set(B(0, 1), B(1, 1), B(2, 1), B(3, 1));
					reg b2 = Simd::set(B(0, 2), B(1, 2), B(2, 2), B(3, 2));
					reg b3 = Simd::set(B(0, 3), B(1, 3), B(2, 3), B(3, 3));

					reg row0 = Simd::loadu(&data[0]);
					reg r0x = Simd::set1(Simd::extract(row0, 0));
					reg r0y = Simd::set1(Simd::extract(row0, 1));
					reg r0z = Simd::set1(Simd::extract(row0, 2));
					reg r0w = Simd::set1(Simd::extract(row0, 3));

					reg acc0 = Simd::mul(r0x, b0);
					acc0 = Simd::fmadd(r0y, b1, acc0);
					acc0 = Simd::fmadd(r0z, b2, acc0);
					acc0 = Simd::fmadd(r0w, b3, acc0);
					Simd::storeu(&result(0, 0), acc0);

					reg row1 = Simd::loadu(&data[4]);
					reg r1x = Simd::set1(Simd::extract(row1, 0));
					reg r1y = Simd::set1(Simd::extract(row1, 1));
					reg r1z = Simd::set1(Simd::extract(row1, 2));
					reg r1w = Simd::set1(Simd::extract(row1, 3));

					reg acc1 = Simd::mul(r1x, b0);
					acc1 = Simd::fmadd(r1y, b1, acc1);
					acc1 = Simd::fmadd(r1z, b2, acc1);
					acc1 = Simd::fmadd(r1w, b3, acc1);
					Simd::storeu(&result(1, 0), acc1);

					reg row2 = Simd::loadu(&data[8]);
					reg r2x = Simd::set1(Simd::extract(row2, 0));
					reg r2y = Simd::set1(Simd::extract(row2, 1));
					reg r2z = Simd::set1(Simd::extract(row2, 2));
					reg r2w = Simd::set1(Simd::extract(row2, 3));

					reg acc2 = Simd::mul(r2x, b0);
					acc2 = Simd::fmadd(r2y, b1, acc2);
					acc2 = Simd::fmadd(r2z, b2, acc2);
					acc2 = Simd::fmadd(r2w, b3, acc2);
					Simd::storeu(&result(2, 0), acc2);

					reg row3 = Simd::loadu(&data[12]);
					reg r3x = Simd::set1(Simd::extract(row3, 0));
					reg r3y = Simd::set1(Simd::extract(row3, 1));
					reg r3z = Simd::set1(Simd::extract(row3, 2));
					reg r3w = Simd::set1(Simd::extract(row3, 3));

					reg acc3 = Simd::mul(r3x, b0);
					acc3 = Simd::fmadd(r3y, b1, acc3);
					acc3 = Simd::fmadd(r3z, b2, acc3);
					acc3 = Simd::fmadd(r3w, b3, acc3);
					Simd::storeu(&result(3, 0), acc3);

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

					alignas(32) K tmp_lo[8], tmp_hi[8];
					reg col_lo[16], col_hi[16];
					for (int j = 0; j < 16; ++j) {
						for (int r = 0; r < 8; ++r)
							tmp_lo[r] = B.data[r * 16 + j];
						for (int r = 0; r < 8; ++r)
							tmp_hi[r] = B.data[(r + 8) * 16 + j];

						col_lo[j] = Simd::loadu(tmp_lo);
						col_hi[j] = Simd::loadu(tmp_hi);
					}

					for (int i = 0; i < 16; ++i) {
						const K* a = &data[i * 16];

						reg acc_lo = Simd::mul( Simd::set1(a[0]), col_lo[0] );
						reg acc_hi = Simd::mul( Simd::set1(a[0]), col_hi[0] );

						acc_lo = Simd::fmadd( Simd::set1(a[1]),  col_lo[1],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[1]),  col_hi[1],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[2]),  col_lo[2],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[2]),  col_hi[2],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[3]),  col_lo[3],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[3]),  col_hi[3],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[4]),  col_lo[4],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[4]),  col_hi[4],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[5]),  col_lo[5],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[5]),  col_hi[5],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[6]),  col_lo[6],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[6]),  col_hi[6],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[7]),  col_lo[7],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[7]),  col_hi[7],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[8]),  col_lo[8],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[8]),  col_hi[8],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[9]),  col_lo[9],  acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[9]),  col_hi[9],  acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[10]), col_lo[10], acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[10]), col_hi[10], acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[11]), col_lo[11], acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[11]), col_hi[11], acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[12]), col_lo[12], acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[12]), col_hi[12], acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[13]), col_lo[13], acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[13]), col_hi[13], acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[14]), col_lo[14], acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[14]), col_hi[14], acc_hi );
						acc_lo = Simd::fmadd( Simd::set1(a[15]), col_lo[15], acc_lo );
						acc_hi = Simd::fmadd( Simd::set1(a[15]), col_hi[15], acc_hi );

						Simd::storeu(&result.data[i * 16 +   0], acc_lo);
						Simd::storeu(&result.data[i * 16 +   8], acc_hi);
					}

					return result;
				}

		};
}
