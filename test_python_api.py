import sys
sys.path.append("pybuild")
from morpheus import Vector, add, sub, scl

from morpheus import Vector, add, sub, scl
import morpheus

v = morpheus.Vector([1.0, 2.0, 3.0])
print(v)
print("len =", len(v))

v2 = morpheus.Vector([4.0, 5.0, 6.0])
v3 = morpheus.add(v, v2)
print("add =", v3)

v4 = morpheus.scl(v, 2.0)
print("scaled =", v4)

a = Vector([1.0, 2.0, 3.0])
b = Vector([4.0, 5.0, 6.0])
print("a =", a)
print("b =", b)

c = add(a, b)
print("a + b =", c)

d = scl(a, 2.5)
print("a * 2.5 =", d)
