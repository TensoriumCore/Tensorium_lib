#include "../includes/Morpheus/Morpheus.hpp"

void saxpy(int n, float *A, float *B, float k)
{
    #pragma morpheus restrict(A,B)
    for (int i = 0; i < n; ++i)
        B[i] += k * A[i];  
}
int main() {
#pragma morpheus dispatch
	int n = 10;
	float A = 0.2;
	float B = 0.22;
	float k = 12;
	
	saxpy(n, &A, &B, k);

    return 0;
}

