from sympy import cos, sin, diff, pi
from sympy.physics.vector import ReferenceFrame, curl, gradient

R = ReferenceFrame("R")

x, y, z = R.varlist
i, j, k = (R.x, R.y, R.z)


def compute_source(u, p, R):
    """
    Given a velocity u and pressure p, compute the body force
    """

    u = u.to_matrix(R)

    J = u.jacobian(R.varlist)
    J += J.T  # symmetrize, divide by 2, times 2 mu with mu = 1

    g = -diff(J[:, 0], x)
    g -= diff(J[:, 1], y)
    g -= diff(J[:, 2], z)

    g += gradient(p, R).to_matrix(R)

    return g.simplify()


## -------------------------------------------------------------------##
"""
The two-dimensional polynomial case
"""

u = i * (-2 * x * y * (x - 1) * (y - 1) * x * (x - 1) * (2 * y - 1))
u += j * (2 * x * y * (x - 1) * (y - 1) * y * (y - 1) * (2 * x - 1))

# p = x * (1 - x) * y * (1 - y)
p = 0

g = compute_source(u, p, R)

print("2D source: ", g)

## -------------------------------------------------------------------##
"""
The three-dimensional case
Generate u = curl phi with phi = 0 on the boundary
"""

phi = i * (1 - x) * x * (1 - cos(2 * pi * y)) * (1 - cos(2 * pi * z))
phi += j * (1 - y) * y * (1 - cos(2 * pi * z)) * (1 - cos(2 * pi * x))
phi += k * (1 - z) * z * (1 - cos(2 * pi * x)) * (1 - cos(2 * pi * y))

u = curl(phi, R)
p = x * (1 - x) * (1 - cos(2 * pi * y)) * sin(2 * pi * z)

g = compute_source(u, p, R)

print("3D source: ", g)
