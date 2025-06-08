#pragma once

#include "../includes/Tensorium/Tensorium.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <random>

int deriv_test();
int linear_solver_test();
int matrix_tests();
int tensor_test();
int vector_tests();
int deriv_test_spectral_fft();
