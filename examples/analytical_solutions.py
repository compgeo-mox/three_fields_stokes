from sympy import cos, sin, diff, pi
from sympy.physics.vector import ReferenceFrame, curl, gradient
from sympy.printing.pycode import NumPyPrinter

def to_numpy(expr):
    return NumPyPrinter().doprint(expr).replace("numpy", "np").replace("R_", "")

def compute_source(u, p, R):
    """
    Given a velocity u and pressure p, compute the body force
    """

    u = u.to_matrix(R)

    J = u.jacobian(R.varlist)
    J += J.T  # symmetrize, divide by 2, times 2 mu with mu = 1

    x, y, z = R.varlist
    g = -diff(J[:, 0], x)
    g -= diff(J[:, 1], y)
    g -= diff(J[:, 2], z)

    g += gradient(p, R).to_matrix(R)

    return g.simplify()


## -------------------------------------------------------------------##
def two_dim(R):
    """
    The two-dimensional polynomial case
    """

    x, y, z = R.varlist
    i, j, k = (R.x, R.y, R.z)

    u = i * (-2 * x * y * (x - 1) * (y - 1) * x * (x - 1) * (2 * y - 1))
    u += j * (2 * x * y * (x - 1) * (y - 1) * y * (y - 1) * (2 * x - 1))

    p = x * (1 - x) * y * (1 - y)
    g = compute_source(u, p, R)
    r = curl(u, R).simplify()

    print("pressure\n", to_numpy(p))

    print("velocity\n", to_numpy(u.to_matrix(R)[0]))
    print("velocity\n", to_numpy(u.to_matrix(R)[1]))
    print("velocity\n", to_numpy(u.to_matrix(R)[2]))

    print("source\n", to_numpy(g[0]))
    print("source\n", to_numpy(g[1]))
    print("source\n", to_numpy(g[2]))

    print("vorticity\n", to_numpy(r.to_matrix(R)[2]))

## -------------------------------------------------------------------##
def three_dim(R):
    """
    The three-dimensional case
    Generate u = curl phi with phi = 0 on the boundary
    """

    x, y, z = R.varlist
    i, j, k = (R.x, R.y, R.z)

    phi = i * (1 - x) * x * (1 - y)**2 * y**2 * (1 - z)**2 * z**2
    #phi += j * (1 - y) * y * (1 - z)**2 * z**2 * (1 - x)**2 * x**2
    #phi += k * (1 - z) * z * (1 - x)**2 * x**2 * (1 - y)**2 * y**2

    u = curl(phi, R)
    p = x * (1 - x) * y * (1 - y) * z * (1 - z)
    g = compute_source(u, p, R)
    r = curl(u, R).simplify()

    print("velocity\n", to_numpy(u.to_matrix(R)[0]))
    print("velocity\n", to_numpy(u.to_matrix(R)[1]))
    print("velocity\n", to_numpy(u.to_matrix(R)[2]))

    print("pressure\n", to_numpy(p))

    print("source\n", to_numpy(g[0]))
    print("source\n", to_numpy(g[1]))
    print("source\n", to_numpy(g[2]))

    print("vorticity\n", to_numpy(r.to_matrix(R)[0]))
    print("vorticity\n", to_numpy(r.to_matrix(R)[1]))
    print("vorticity\n", to_numpy(r.to_matrix(R)[2]))

if __name__ == "__main__":
    R = ReferenceFrame("R")

    two_dim(R)
    #three_dim(R)
