#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

using namespace tensorium;

REGISTER_TEST("core.linear.gauss", "Gaussian elimination solver", []() {
    Matrix<float> A(2, 2);
    A(0, 0) = 2.0f; A(0, 1) = 1.0f;
    A(1, 0) = 5.0f; A(1, 1) = 7.0f;
    Vector<float> b = {11.0f, 13.0f};

    Vector<float> x = gauss_solve(A, b);
    Vector<float> b_check = A * x;
    tensorium::tests::expect_near(b_check[0], b[0], 1e-4f, "gauss solve b0");
    tensorium::tests::expect_near(b_check[1], b[1], 1e-4f, "gauss solve b1");
});

REGISTER_TEST("core.linear.jacobi", "Jacobi solver and row-echelon rank", []() {
    Matrix<float> A2(3, 3);
    A2(0, 0) = 10.0f; A2(0, 1) = -1.0f; A2(0, 2) = 2.0f;
    A2(1, 0) = -1.0f; A2(1, 1) = 11.0f; A2(1, 2) = -1.0f;
    A2(2, 0) = 2.0f;  A2(2, 1) = -1.0f; A2(2, 2) = 10.0f;
    Vector<float> b2 = {6.0f, 25.0f, -11.0f};

    Vector<float> x2 = jacobi_solve(A2, b2);
    Vector<float> check = mul_vec(A2, x2);
    tensorium::tests::expect_near(check[0], b2[0], 1e-3f, "jacobi b0");
    tensorium::tests::expect_near(check[1], b2[1], 1e-3f, "jacobi b1");
    tensorium::tests::expect_near(check[2], b2[2], 1e-3f, "jacobi b2");

    Matrix<float> A3(3, 3);
    A3(0, 0) = 1.0f;  A3(0, 1) = 2.0f;  A3(0, 2) = -1.0f;
    A3(1, 0) = 2.0f;  A3(1, 1) = 4.0f;  A3(1, 2) = -2.0f;
    A3(2, 0) = -1.0f; A3(2, 1) = -2.0f; A3(2, 2) = 1.0f;

    Vector<float> b3 = {3.0f, 6.0f, -3.0f};
    row_echelon(A3, &b3);
    size_t rank = rank_mat(A3);
    TENSORIUM_TEST_ASSERT(rank == 1);
});

REGISTER_TEST("core.linear.jacobi_double", "Jacobi solver with double precision", []() {
    Matrix<double> A(2, 2);
    A(0, 0) = 5.0;
    A(1, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 1) = 4.0;
    Vector<double> b = {7.0, 6.0};
    auto x = solver::Jacobi<double>::solve(A, b, 1e-12, 500);
    auto Ax = A.mul_vec(x);
    tensorium::tests::expect_near(Ax[0], b[0], 1e-9, "jacobi double b0");
    tensorium::tests::expect_near(Ax[1], b[1], 1e-9, "jacobi double b1");
});
