#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <random>

using namespace tensorium;

REGISTER_TEST("core.matrix.arithmetic", "Matrix arithmetic and inversion", []() {
    using Mat = Matrix<float>;
    using Vec = Vector<float>;

    Mat A(2, 2);
    A(0, 0) = 1.0f;
    A(0, 1) = 2.0f;
    A(1, 0) = 3.0f;
    A(1, 1) = 4.0f;

    Mat B(2, 2);
    B(0, 0) = 5.0f;
    B(0, 1) = 6.0f;
    B(1, 0) = 7.0f;
    B(1, 1) = 8.0f;

    Mat C = A;
    C.add(B);
    tensorium::tests::expect_near(C(0, 0), 6.0f, 1e-4, "matrix add");
    tensorium::tests::expect_near(C(1, 1), 12.0f, 1e-4, "matrix add");

    C.sub(B);
    tensorium::tests::expect_near(C(0, 0), A(0, 0), 1e-4, "matrix sub");

    C.scl(2.0f);
    tensorium::tests::expect_near(C(0, 0), 2.0f * A(0, 0), 1e-4, "matrix scale");

    Vec x = {1, 1};
    Vec y = A.mul_vec(x);
    tensorium::tests::expect_near(y[0], 3.0f, 1e-4, "matrix mul_vec");
    tensorium::tests::expect_near(y[1], 7.0f, 1e-4, "matrix mul_vec");

    Mat T = A.transpose();
    tensorium::tests::expect_near(T(0, 1), A(1, 0), 1e-4, "matrix transpose");

    Mat S = A;
    S.swap_rows(0, 1);
    tensorium::tests::expect_near(S(0, 0), A(1, 0), 1e-4, "matrix swap");

    Mat tr = A.trace();
    tensorium::tests::expect_near(tr(0, 0), A(0, 0) + A(1, 1), 1e-4, "matrix trace");

    Mat M = tensorium::mul_mat(A, B);
    tensorium::tests::expect_near(M(0, 0), 1 * 5 + 2 * 7, 1e-4, "matrix mul");
    tensorium::tests::expect_near(M(1, 1), 3 * 6 + 4 * 8, 1e-4, "matrix mul");

    A(0, 0) = 4.0f;
    A(0, 1) = 7.0f;
    A(1, 0) = 2.0f;
    A(1, 1) = 6.0f;

    auto A_inv = inverse_mat(A);
    tensorium::tests::expect_near(A_inv(0, 0), 0.6f, 1e-3f, "matrix inverse");
    tensorium::tests::expect_near(A_inv(0, 1), -0.7f, 1e-3f, "matrix inverse");
    tensorium::tests::expect_near(A_inv(1, 0), -0.2f, 1e-3f, "matrix inverse");
    tensorium::tests::expect_near(A_inv(1, 1), 0.4f, 1e-3f, "matrix inverse");

    Mat Id = tensorium::mul_mat(A, A_inv);
    tensorium::tests::expect_near(Id(0, 0), 1.0f, 1e-3f, "matrix identity");
    tensorium::tests::expect_near(Id(1, 1), 1.0f, 1e-3f, "matrix identity");
    tensorium::tests::expect_near(Id(0, 1), 0.0f, 1e-3f, "matrix identity");
    tensorium::tests::expect_near(Id(1, 0), 0.0f, 1e-3f, "matrix identity");

    float det = det_mat(A);
    tensorium::tests::expect_near(det, 4 * 6 - 7 * 2, 1e-5f, "matrix det");
});

