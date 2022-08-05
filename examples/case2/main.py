import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
from stokes import Stokes2DHyb, Stokes
import error

def vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]

    first = 24*x**4*y - 12*x**4 - 48*x**3*y + 24*x**3 + 48*x**2*y**3 - 72*x**2*y**2 + 48*x**2*y - 12*x**2 - 48*x*y**3 + 74*x*y**2 - 26*x*y + 8*y**3 - 13*y**2 + 5*y

    second = -48*x**3*y**2 + 48*x**3*y - 8*x**3 + 72*x**2*y**2 - 70*x**2*y + 11*x**2 - 24*x*y**4 + 48*x*y**3 - 48*x*y**2 + 22*x*y - 3*x + 12*y**4 - 24*y**3 + 12*y**2

    source = np.vstack((first, second, np.zeros(sd.num_faces)))
    return np.sum(sd.face_normals * source, axis=0)

def r_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    return 12*x**2*y**2*(x - 1)**2 + 12*x**2*y**2*(y - 1)**2 - 12*x**2*y*(x - 1)**2 + 2*x**2*(x - 1)**2 - 12*x*y**2*(y - 1)**2 + 2*y**2*(y - 1)**2

def q_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    first = -2 * x * y * (x - 1) * (y - 1) * x * (x - 1) * (2 * y - 1)
    second = 2 * x * y * (x - 1) * (y - 1) * y * (y - 1) * (2 * x - 1)
    return np.vstack((first, second, np.zeros(sd.num_cells)))

def p_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    return  x*y*(1 - x)*(1 - y)

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

def solve_stokes(mdg, keyword, source, bc_val):
    st = Stokes(keyword)
    spp, rhs = st.matrix_and_rhs(mdg, source, bc_val)

    # solve the problem
    x = sps.linalg.spsolve(spp, rhs)
    r = x[:mdg.num_subdomain_ridges()]
    q = x[mdg.num_subdomain_ridges():mdg.num_subdomain_ridges()+mdg.num_subdomain_faces()]
    p = x[-mdg.num_subdomain_cells():]

    return r, q, p

def solve_stokes_hyb(mdg, keyword, source, bc_val):
    # create the Stokes solver
    st = Stokes2DHyb(keyword)
    spp, rhs, M, curl = st.matrix_and_rhs(mdg, source, bc_val)

    # solve the problem
    x = sps.linalg.spsolve(spp, rhs)
    q = x[:mdg.num_subdomain_faces()]
    p = x[-mdg.num_subdomain_cells():]

    # post process vorticity
    r = sps.linalg.spsolve(M, curl.T @ q)
    return r, q, p

def main(mdg, stokes, ne, keyword = "flow"):
    # set the data
    bc_val = []
    source = []
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        bc_val.append(np.zeros(sd.num_faces))
        source.append(vector_source(sd))

    r, q, p = stokes(mdg, keyword, np.hstack(source), np.hstack(bc_val))

    # post process vorticity
    proj_r = pg.eval_at_cell_centers(mdg, ne(keyword))
    P0r = proj_r * r

    # post process
    proj_q = pg.proj_faces_to_cells(mdg)
    P0q = (proj_q * q).reshape((3, -1), order="F")

    # compute the error
    return error.compute(sd, r, r_ex, P0q, q_ex, p, p_ex)

    #for sd, data in mdg.subdomains(return_data=True):
    #   data[pp.STATE] = {"P0r": P0r, "P0q": P0q, "p": p, "err_p": p_ex(sd) - p, "p_ex": p_ex(sd)}

    #save = pp.Exporter(mdg, "sol", folder_name=folder)
    #save.write_vtu(["P0r", "P0q", "p"])

if __name__ == "__main__":

    N = 2 ** np.arange(4, 9)
    print("stokes")
    err = np.array([main(create_grid(n), solve_stokes, pg.Lagrange) for n in N])
    error.order(err)

    print("stokes hyb")
    err_hyb = np.array([main(create_grid(n), solve_stokes_hyb, pg.Lagrange) for n in N])
    error.order(err_hyb)
