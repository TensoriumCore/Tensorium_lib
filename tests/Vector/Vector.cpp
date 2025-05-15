#include "../../includes/Tensorium/Tensorium.hpp"

using namespace tensorium;

#define CHECK(expr) \
	do { \
		if (!(expr)) { \
			std::cerr << "❌ CHECK FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
			std::exit(1); \
		} \
	} while (0)

int vector_tests() {
	using Vec = Vector<float>;

	try {
		Vec v1 = {1, 2, 3, 4};
		CHECK(v1.size() == 4);
		CHECK(v1[0] == 1);
		
		Vec v2 = {4, 3, 2, 1};
		v1.add(v2);
		CHECK(v1[0] == 5 && v1[1] == 5);

		v1.sub(v2);
		CHECK(v1[0] == 1 && v1[1] == 2);

		v1.scl(2.0f);
		CHECK(v1[0] == 2 && v1[1] == 4);

		Vec diff = v2 - v1;
		CHECK(diff[0] == 2.0f);

		float dot = v1.dot(v2);
		CHECK(std::abs(dot - (2*4 + 4*3 + 6*2 + 8*1)) < 1e-4);

		CHECK(std::abs(v1.norm_1() - 20.0f) < 1e-4);
		CHECK(std::abs(v1.norm_2() - std::sqrt(120.0f)) < 1e-4);
		CHECK(std::abs(v1.norm_inf() - 8.0f) < 1e-4);

		Vec u = {1, 0, 0};
		Vec v = {0, 1, 0};
		float angle = Vector<float>::angle_cos(u, v);
		CHECK(std::abs(angle) < 1e-6);

		Vec cp = Vector<float>::cross_product(u, v);
		CHECK(cp.size() == 3);
		CHECK(cp[0] == 0 && cp[1] == 0 && cp[2] == 1);

		std::vector<Vec> basis = {
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
		};
		std::vector<float> coefs = {2.0f, -1.0f, 3.5f};
		Vec result = Vector<float>::linear_combination(basis, coefs);
		CHECK(result.size() == 4);
		CHECK(std::abs(result[0] - 2.0f) < 1e-4);
		CHECK(std::abs(result[1] + 1.0f) < 1e-4);
		CHECK(std::abs(result[2] - 3.5f) < 1e-4);

		Vec a = {0, 0, 0, 0};
		Vec b = {1, 1, 1, 1};
		Vec mid = Vector<float>::lerp(a, b, 0.5f);
		for (size_t i = 0; i < mid.size(); ++i)
			CHECK(std::abs(mid[i] - 0.5f) < 1e-5);

	} catch (const std::exception& e) {
		std::cerr << "❌ EXCEPTION: " << e.what() << "\n";
		return 1;
	}

	std::cout << "✅ All Vector tests passed.\n";
	return 0;
}
