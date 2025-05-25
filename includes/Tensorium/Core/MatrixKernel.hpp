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
				inline Matrix<K> mul_mat3x3(const Matrix<K>& mat) const {
					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg  = typename Simd::reg;

					Matrix<K> result(3, 3);
					reg row0 = Simd::loadu(&data[0]);
					reg row1 = Simd::loadu(&data[3]);
					reg row2 = Simd::loadu(&data[6]);

					for (int i = 0; i < 3; ++i) {
						reg row = (i==0 ? row0 : (i==1 ? row1 : row2));

						K x = data[i * 3 + 0];
						K y = data[i * 3 + 1];
						K z = data[i * 3 + 2];

						reg sx = Simd::set1(x);
						reg sy = Simd::set1(y);
						reg sz = Simd::set1(z);

						reg c0 = Simd::loadu(&mat.data[0]);
						reg c1 = Simd::loadu(&mat.data[3]);
						reg c2 = Simd::loadu(&mat.data[6]);

						reg acc = Simd::mul(sx, c0);
						acc = Simd::fmadd(sy, c1, acc);
						acc = Simd::fmadd(sz, c2, acc);

						Simd::storeu(&result.data[i*3], acc);
					}

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

		};
}
