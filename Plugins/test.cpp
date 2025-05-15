#include "../includes/Tensorium/Tensorium.hpp"
// #include <iostream>

float* f() {
    float *a = new float[8]; // force une allocation
    a[0] = 1.0f;
    return a;
}
// void saxpy(int n, float *A, float *B, float k)
// {
//     #pragma tensorium restrict(A,B)  
//     for (int i = 0; i < n; ++i)
//         B_re[i] += k * A_re[i];  
// }

int main()
{
#pragma tensorium dispatch
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
	//
	tensorium::Matrix<float> A2(2, 2);
	tensorium::Matrix<float> B2(2, 2);
	A2(0, 0) = 1.0f; A2(0, 1) = 2.0f;
	A2(1, 0) = 3.0f; A2(1, 1) = 4.0f;
	B2(0, 0) = 5.0f; B2(0, 1) = 6.0f;
	B2(1, 0) = 7.0f; B2(1, 1) = 8.0f;
	auto C2 = tensorium::mul_mat(A2, B2);
	C2.print();
    return 0;
}


