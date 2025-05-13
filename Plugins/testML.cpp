
#include <immintrin.h>
#include <iostream>

int main() {
    alignas(32) float a[8] = {1,2,3,4,5,6,7,8};
    alignas(32) float b[8] = {8,7,6,5,4,3,2,1};
    alignas(32) float result[8];

    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);
    __m256 vr = _mm256_add_ps(va, vb);

    _mm256_store_ps(result, vr);

    for (int i = 0; i < 8; ++i)
        std::cout << result[i] << ' ';
    std::cout << '\n';

    return 0;
}
