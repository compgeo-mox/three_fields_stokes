import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys

sys.path.append("../../src/")
from stokes import Stokes3D


def vector_source(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = 0
    second = 4*np.pi*(2*x*np.pi**2*(x - 1)*(np.cos(2*y*np.pi) - 1) + 2*x*np.pi**2*(x - 1)*np.cos(2*y*np.pi) - np.cos(2*y*np.pi) + 1)*np.sin(2*z*np.pi)
    third = 4*np.pi*(-2*x*np.pi**2*(x - 1)*(np.cos(2*z*np.pi) - 1) - 2*x*np.pi**2*(x - 1)*np.cos(2*z*np.pi) + np.cos(2*z*np.pi) - 1)*np.sin(2*y*np.pi)

    return np.vstack(
        (sd.cell_volumes * first, sd.cell_volumes * second, sd.cell_volumes * third)
    ).ravel(order="F")


def p_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    return 0 #x * (1 - x) * (1 - np.cos(2 * y * np.pi)) * np.sin(2 * z * np.pi)


def u_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = np.zeros(sd.num_cells)
    second = 2*x*np.pi*(1 - x)*(1 - np.cos(2*y*np.pi))*np.sin(2*z*np.pi)
    third = -2*x*np.pi*(1 - x)*(1 - np.cos(2*z*np.pi))*np.sin(2*y*np.pi)

    return np.vstack((first, second, third))


def create_grid(n):
    # make the grid
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    network = pp.FractureNetwork3d(domain=domain)

    mesh_size = 1/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg


def main(mdg, keyword="flow"):
    # set the data
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

    vect = np.hstack([vector_source(sd) for sd in mdg.subdomains()])

    # create the Stokes solver
    st = Stokes3D(keyword)
    print("discretize the matrix and rhs")
    spp, rhs, proj = st.matrix_and_rhs(mdg, vect)

    # solve the problem
    print("solve the problem")
    x = sps.linalg.spsolve(spp, rhs.tocsc())
    print("done")
    u = x[:mdg.num_subdomain_faces()]
    p = x[-mdg.num_subdomain_cells():]

    # post process
    P0u = (proj.T * u).reshape((3, -1), order="F")

    err_u = np.sqrt(
        np.trace((u_ex(sd) - P0u) @ sps.diags(sd.cell_volumes) @ (u_ex(sd) - P0u).T)
    )
    err_p = np.sqrt((p_ex(sd) - p) @ sps.diags(sd.cell_volumes) @ (p_ex(sd) - p).T)

    print(spp.shape[0], np.mean(sd.cell_diameters()), err_u, err_p)

    # for sd, data in mdg.subdomains(return_data=True):
    #    data[pp.STATE] = {"P0u": P0u, "p": p}

    # save = pp.Exporter(mdg, "sol")
    # save.write_vtu(["P0u", "p"])


if __name__ == "__main__":
    N = 2**np.arange(3, 7)
    [main(create_grid(n)) for n in N]
