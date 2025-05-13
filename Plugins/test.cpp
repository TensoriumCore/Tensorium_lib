// #include "../includes/Morpheus/Morpheus.hpp"
// #include <iostream>

float* f() {
    float *a = new float[8]; // force une allocation
    a[0] = 1.0f;
    return a;
}
// void saxpy(int n, float *A, float *B, float k)
// {
//     #pragma morpheus restrict(A,B)  
//     for (int i = 0; i < n; ++i)
//         B_re[i] += k * A_re[i];  
// }

int main()
{
#pragma morpheus dispatch
    const int n = 10;
    float A[n], B[n];
    for (int i = 0; i < n; ++i) {
        A[i] = 0.2f * i;
        B[i] = 0.22f * i;
    }

    float k = 12.0f;
    // saxpy(n, A, B, k);
	f();
    // for (float v : B) std::cout << v << ' ';
    // std::cout << '\n';
    return 0;
}


