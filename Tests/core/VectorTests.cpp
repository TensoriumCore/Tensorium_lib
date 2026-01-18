#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Tensorium.hpp"

#include <vector>

using namespace tensorium;

REGISTER_TEST("core.vector.basic", "Vector utility operations", []() {
    using Vec = Vector<float>;

    Vec v1 = {1, 2, 3, 4};
    TENSORIUM_TEST_ASSERT(v1.size() == 4);
    TENSORIUM_TEST_ASSERT(v1[0] == 1);

    Vec v2 = {4, 3, 2, 1};
    v1.add(v2);
    TENSORIUM_TEST_ASSERT(v1[0] == 5 && v1[1] == 5);

    v1.sub(v2);
    TENSORIUM_TEST_ASSERT(v1[0] == 1 && v1[1] == 2);

    v1.scl(2.0f);
    TENSORIUM_TEST_ASSERT(v1[0] == 2 && v1[1] == 4);

    Vec diff = v2 - v1;
    TENSORIUM_TEST_ASSERT(diff[0] == 2.0f);

    float dot = v1.dot(v2);
    tensorium::tests::expect_near(dot, 2 * 4 + 4 * 3 + 6 * 2 + 8 * 1, 1e-4,
                                  "vector dot product");

    tensorium::tests::expect_near(v1.norm_1(), 20.0f, 1e-4, "vector L1 norm");
    tensorium::tests::expect_near(v1.norm_2(), std::sqrt(120.0f), 1e-4, "vector L2 norm");
    tensorium::tests::expect_near(v1.norm_inf(), 8.0f, 1e-4, "vector Linf norm");

    Vec u = {1, 0, 0};
    Vec v = {0, 1, 0};
    float angle = Vector<float>::angle_cos(u, v);
    tensorium::tests::expect_near(angle, 0.0f, 1e-6, "vector angle");

    Vec cp = Vector<float>::cross_product(u, v);
    TENSORIUM_TEST_ASSERT(cp.size() == 3);
    TENSORIUM_TEST_ASSERT(cp[0] == 0 && cp[1] == 0 && cp[2] == 1);

    std::vector<Vec> basis = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    std::vector<float> coefs = {2.0f, -1.0f, 3.5f};
    Vec result = Vector<float>::linear_combination(basis, coefs);
    TENSORIUM_TEST_ASSERT(result.size() == 4);
    tensorium::tests::expect_near(result[0], 2.0f, 1e-4, "vector comb x");
    tensorium::tests::expect_near(result[1], -1.0f, 1e-4, "vector comb y");
    tensorium::tests::expect_near(result[2], 3.5f, 1e-4, "vector comb z");

    Vec a = {0, 0, 0, 0};
    Vec b = {1, 1, 1, 1};
    Vec mid = Vector<float>::lerp(a, b, 0.5f);
    for (size_t i = 0; i < mid.size(); ++i) {
        tensorium::tests::expect_near(mid[i], 0.5f, 1e-5, "vector lerp");
    }
});

REGISTER_TEST("core.vector.double_precision", "Vector<double> preserves tiny scalars", []() {
    Vector<double> v(1, 1.0);
    const double scale = 1e-15;
    v.scl(scale);
    tensorium::tests::expect_near(v[0], scale, 1e-20, "vector double precision");
});

REGISTER_TEST("core.vector.double_ops", "Double precision dot, norm, lerp", []() {
    Vector<double> a = {1e-12, -2e-12, 3e-12, -4e-12};
    double dot = a.dot(a);
    tensorium::tests::expect_near(dot, 30e-24, 1e-30, "dot double");
    double norm = a.norm_2();
    tensorium::tests::expect_near(norm, std::sqrt(dot), 1e-30, "norm double");
    Vector<double> b = {-1.0, 2.0, -3.0, 4.0};
    Vector<double> c = Vector<double>::lerp(a, b, 0.25);
    tensorium::tests::expect_near(c[0], 0.25 * b[0] + 0.75 * a[0], 1e-15, "lerp double 0");
    tensorium::tests::expect_near(c[3], 0.25 * b[3] + 0.75 * a[3], 1e-15, "lerp double 3");
});
