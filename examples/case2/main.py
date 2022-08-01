import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
from stokes import Stokes2D

def vector_source(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    first = -4 * (
        x**4 * (6 * y - 3)
        + x**3 * (6 - 12 * y)
        + 3 * x**2 * (4 * y**3 - 6 * y**2 + 4 * y - 1)
        - 6 * x * y * (2 * y**2 - 3 * y + 1)
        + y * (2 * y**2 - 3 * y + 1)
    )
    second = 4 * (
        2 * x**3 * (6 * y**2 - 6 * y + 1)
        - 3 * x**2 * (6 * y**2 - 6 * y + 1)
        + x * (6 * y**4 - 12 * y**3 + 12 * y**2 - 6 * y + 1)
        - 3 * (y - 1) ** 2 * y**2
    )
    return np.vstack(
        (-sd.cell_volumes * first, -sd.cell_volumes * second, np.zeros(sd.num_cells))
    ).ravel(order="F")


def p_ex(sd):
    return np.zeros(sd.num_cells)


def u_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    first = -2 * x * y * (x - 1) * (y - 1) * x * (x - 1) * (2 * y - 1)
    second = 2 * x * y * (x - 1) * (y - 1) * y * (y - 1) * (2 * x - 1)
    return np.vstack((first, second, np.zeros(sd.num_cells)))


def create_grid(n):
    # make the grid
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    network = pp.FractureNetwork2d(domain=domain)

    mesh_size = 1/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

def main(mdg, keyword = "flow"):
    # set the data
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

    vect = np.hstack([vector_source(sd) for sd in mdg.subdomains()])

    # create the Stokes solver
    st = Stokes2D(keyword)
    spp, rhs, proj = st.matrix_and_rhs(mdg, vect)

    # solve the problem
    x = sps.linalg.spsolve(spp, rhs.tocsc())
    u = x[:mdg.num_subdomain_faces()]
    p = x[-mdg.num_subdomain_cells():]

    # post process
    P0u = (proj.T * u).reshape((3, -1), order="F")

    err_u = np.sqrt(
        np.trace((u_ex(sd) - P0u) @ sps.diags(sd.cell_volumes) @ (u_ex(sd) - P0u).T)
    )
    err_p = np.sqrt((p_ex(sd) - p) @ sps.diags(sd.cell_volumes) @ (p_ex(sd) - p).T)

    print(np.mean(sd.cell_diameters()), err_u, err_p)

    # save = pp.Exporter(g, "sol")
    # save.write_vtu([("P0u", P0u), ("p", p), ("u_ex", u_ex(g))])


if __name__ == "__main__":
    N = 2 ** np.arange(3, 8)
    [main(create_grid(n)) for n in N]
