"""Basic usage example for tensorium Python bindings."""
from tensorium.vectors import Vector, add as add_vec, dot
from tensorium.matrices import Matrix, add as add_mat


def main() -> None:
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    print("v1 + v2 =", add_vec(v1, v2))
    print("dot(v1, v2) =", dot(v1, v2))

    m1 = Matrix(2, 2)
    m1.fill([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix(2, 2)
    m2.fill([[5.0, 6.0], [7.0, 8.0]])
    print("m1 + m2 =")
    add_mat(m1, m2).print()


if __name__ == "__main__":
    main()
