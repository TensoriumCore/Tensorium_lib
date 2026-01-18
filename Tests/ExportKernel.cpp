#include "../includes/Tensorium/Core/MatrixKernels/MatrixKernel.hpp"

using K = float;

namespace tensorium {
    template class MatrixKernel<K>;
}

extern "C" __attribute__((noinline, used, hot, annotate("tensorium_dump")))
void TENSORIUM_DUMP_mul_mat2x2_f(tensorium::MatrixKernel<K>* A,
                                 tensorium::MatrixKernel<K>* B,
                                 tensorium::Matrix<K>* C) {
    *C = A->mul_mat32x32(*B);
}


