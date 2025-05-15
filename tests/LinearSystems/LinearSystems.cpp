#include "../test.hpp"

int linear_solver_test() {
	using tensorium::Matrix;
	using tensorium::Vector;

	std::cout << "=== Gauss Solver Test ===\n";
	Matrix<float> A(2, 2);
	A(0, 0) = 2.0f; A(0, 1) = 1.0f;
	A(1, 0) = 5.0f; A(1, 1) = 7.0f;
	Vector<float> b = { 11.0f, 13.0f };

	Vector<float> x = tensorium::gauss_solve(A, b);
	std::cout << "Solution x:\n";
	x.print();

	Vector<float> b_check = A * x;
	std::cout << "Check Ax = b:\n";
	b_check.print();

	std::cout << "=== Jacobi Solver Test ===\n";
	Matrix<float> A2(3, 3);
	A2(0, 0) = 10.0f; A2(0, 1) = -1.0f; A2(0, 2) = 2.0f;
	A2(1, 0) = -1.0f; A2(1, 1) = 11.0f; A2(1, 2) = -1.0f;
	A2(2, 0) = 2.0f;  A2(2, 1) = -1.0f; A2(2, 2) = 10.0f;

	Vector<float> b2 = { 6.0f, 25.0f, -11.0f };
	Vector<float> x2 = tensorium::jacobi_solve(A2, b2);
	std::cout << "Solution x2:\n";
	x2.print();

	Vector<float> b2_check = tensorium::mul_vec(A2, x2);
	std::cout << "Check Ax2 = b2:\n";
	b2_check.print();

	std::cout << "=== Row Echelon Test ===\n";
	Matrix<float> A3(3, 3);
	A3(0, 0) = 1.0f;  A3(0, 1) = 2.0f;  A3(0, 2) = -1.0f;
	A3(1, 0) = 2.0f;  A3(1, 1) = 4.0f;  A3(1, 2) = -2.0f;
	A3(2, 0) = -1.0f; A3(2, 1) = -2.0f; A3(2, 2) = 1.0f;

	Vector<float> b3 = { 3.0f, 6.0f, -3.0f };

	tensorium::row_echelon(A3, &b3);

	std::cout << "Echelon form of A3:\n";
	A3.print();

	std::cout << "Modified b3:\n";

	b3.print();
	std::cout << "Rank = " << tensorium::rank_mat(A3) << "\n";
	return 0;
}
