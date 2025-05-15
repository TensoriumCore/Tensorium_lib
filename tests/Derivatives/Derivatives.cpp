#include "../test.hpp"

int deriv_test() {
	std::cout << "\n=== Derivate 2D Test (\u2202/\u2202x) ===\n";
	tensorium::Derivate<float> f2d(4, 4);
	tensorium::Derivate<float> dfdx2d(4, 4);

	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			f2d(i, j) = static_cast<float>(i * 10 + j);

	tensorium::centered_derivative(f2d, dfdx2d, 0, 1.0f);

	std::cout << "\n=== Derivate 2D Test (\u2202/\u2202y) ===\n";
	tensorium::Derivate<float> f2d_y(4, 4);
	tensorium::Derivate<float> dfdx2d_y(4, 4);
	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			f2d_y(i, j) = static_cast<float>(i * 10 + j);
	tensorium::centered_derivative(f2d_y, dfdx2d_y, 1, 1.0f);


	std::cout << "\n=== DerivateND 3D Test (\u2202/\u2202x) ===\n";
	std::array<size_t, 3> dims = {4, 4, 4};
	tensorium::DerivateND<float, 3> fnd_x(dims), dfdxnd(dims);
	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			for (size_t k = 0; k < 4; ++k)
				fnd_x({i, j, k}) = float(i + j + k);
	tensorium::centered_derivative(fnd_x, dfdxnd, 0, 1.0f);


	std::cout << "\n=== DerivateND 3D Test (\u2202/\u2202z) ===\n";
	tensorium::DerivateND<float, 3> fnd(dims), dfdznd(dims);
	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			for (size_t k = 0; k < 4; ++k)
				fnd({i, j, k}) = float(i + j + k);
	tensorium::centered_derivative(fnd, dfdznd, 2, 1.0f);


	tensorium::centered_derivative_order4(fnd, dfdznd, 2, 1.0f);
	std::cout << "\u2202f/\u2202z (ordre 4) slice at k=2:\n";


	const size_t N = 16384;
	const float dx = 0.01f;
	const float pi = 3.14159265358979323846f;

	tensorium::Derivate<float> f(N, 1), df_order2(N, 1), df_order4(N, 1), df_exact(N, 1);
	for (size_t i = 0; i < N; ++i) {
		float x = i * dx;
		f(i, 0) = std::sin(x) + 0.1f * std::sin(10*x);
		df_exact(i, 0) = std::cos(x) + 1.0f * std::cos(10*x);
	}
	f.centered_derivative(f, df_order2, 0, dx);
	f.centered_derivative_order4(f, df_order4, 0, dx);

	float max_err_order2 = 0.0f, max_err_order4 = 0.0f;
	float avg_err_order2 = 0.0f, avg_err_order4 = 0.0f;
	size_t valid = 0;
	for (size_t i = 2; i < N - 2; ++i) {
		float exact = df_exact(i, 0);
		float e2 = std::abs(df_order2(i, 0) - exact);
		float e4 = std::abs(df_order4(i, 0) - exact);
		max_err_order2 = std::max(max_err_order2, e2);
		max_err_order4 = std::max(max_err_order4, e4);
		avg_err_order2 += e2;
		avg_err_order4 += e4;
		++valid;
	}
	avg_err_order2 /= valid;
	avg_err_order4 /= valid;

	std::cout << "Points valides: " << valid << " / " << N << "\n";
	std::cout << "Erreur max (ordre 2): " << max_err_order2 << "\n";
	std::cout << "Erreur max (ordre 4): " << max_err_order4 << "\n";
	std::cout << "Erreur moyenne (ordre 2): " << avg_err_order2 << "\n";
	std::cout << "Erreur moyenne (ordre 4): " << avg_err_order4 << "\n";

	std::cout << "\nV\u00e9rification des bords:\n";
	std::cout << "df_order4(0,0) = " << df_order4(0, 0) << ", attendu: 0\n";
	std::cout << "df_order4(1,0) = " << df_order4(1, 0) << ", attendu: 0\n";
	std::cout << "df_order4(N-2,0) = " << df_order4(N-2, 0) << ", attendu: 0\n";
	std::cout << "df_order4(N-1,0) = " << df_order4(N-1, 0) << ", attendu: 0\n";

	return 0;
}
