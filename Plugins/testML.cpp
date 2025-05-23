#include <immintrin.h>
#include <iostream>
#include <iomanip>

int main() {
    alignas(32) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) float b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    alignas(32) float c[8] = {0};
    alignas(32) float d[8] = {0};

    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);

    __m256 vsum  = _mm256_add_ps(va, vb);
    __m256 vdiff = _mm256_sub_ps(va, vb);
    __m256 vmul  = _mm256_mul_ps(vsum, vdiff);
    __m256 vdiv  = _mm256_div_ps(vmul, vb);

    _mm256_store_ps(c, vmul);
    _mm256_store_ps(d, vdiv);

    std::cout << "Result (mul): ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(8) << c[i] << ' ';
    std::cout << "\n";

    std::cout << "Result (div): ";
    for (int i = 0; i < 8; ++i)
        std::cout << std::setw(8) << d[i] << ' ';
    std::cout << "\n";

    return 0;
}
