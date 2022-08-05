import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
from stokes import Stokes3DHyb, Stokes
import error

def vector_source(sd):
    x = sd.face_centers[0, :]
    y = sd.face_centers[1, :]
    z = sd.face_centers[2, :]

    first = y*z*(1 - 2*x)*(y - 1)*(z - 1)
    second = 4*x*y**2*z**2*(x - 1)*(z - 1) + 12*x*y**2*z*(x - 1)*(y - 1)**2 + 4*x*y**2*z*(x - 1)*(z - 1)**2 + 12*x*y**2*(x - 1)*(y - 1)**2*(z - 1) + 16*x*y*z**2*(x - 1)*(y - 1)*(z - 1) + 16*x*y*z*(x - 1)*(y - 1)*(z - 1)**2 - x*y*z*(x - 1)*(z - 1) + 4*x*z**2*(x - 1)*(y - 1)**2*(z - 1) + 4*x*z*(x - 1)*(y - 1)**2*(z - 1)**2 - x*z*(x - 1)*(y - 1)*(z - 1) + 4*y**2*z**2*(y - 1)**2*(z - 1) + 4*y**2*z*(y - 1)**2*(z - 1)**2
    third = -4*x*y**2*z**2*(x - 1)*(y - 1) - 16*x*y**2*z*(x - 1)*(y - 1)*(z - 1) - 4*x*y**2*(x - 1)*(y - 1)*(z - 1)**2 - 4*x*y*z**2*(x - 1)*(y - 1)**2 - 12*x*y*z**2*(x - 1)*(z - 1)**2 - 16*x*y*z*(x - 1)*(y - 1)**2*(z - 1) - x*y*z*(x - 1)*(y - 1) - 4*x*y*(x - 1)*(y - 1)**2*(z - 1)**2 - x*y*(x - 1)*(y - 1)*(z - 1) - 12*x*z**2*(x - 1)*(y - 1)*(z - 1)**2 - 4*y**2*z**2*(y - 1)*(z - 1)**2 - 4*y*z**2*(y - 1)**2*(z - 1)**2

    source = np.vstack((first, second, third))
    return np.sum(sd.face_normals * source, axis=0)

def r_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = 2*x*(x - 1)*(y**2*z**2*(y - 1)**2 + y**2*z**2*(z - 1)**2 + 4*y**2*z*(y - 1)**2*(z - 1) + y**2*(y - 1)**2*(z - 1)**2 + 4*y*z**2*(y - 1)*(z - 1)**2 + z**2*(y - 1)**2*(z - 1)**2)
    second = -2*y*z**2*(y - 1)*(z - 1)**2*(x*y + x*(y - 1) + y*(x - 1) + (x - 1)*(y - 1))
    third = -2*y**2*z*(y - 1)**2*(z - 1)*(x*z + x*(z - 1) + z*(x - 1) + (x - 1)*(z - 1))

    return np.vstack((first, second, third))

def q_ex(sd):
    x = sd.cell_centers[0, :]
    y = sd.cell_centers[1, :]
    z = sd.cell_centers[2, :]

    first = np.zeros(sd.num_cells)
    second = x*y**2*z**2*(1 - x)*(1 - y)**2*(2*z - 2) + 2*x*y**2*z*(1 - x)*(1 - y)**2*(1 - z)**2
    third = -x*y**2*z**2*(1 - x)*(1 - z)**2*(2*y - 2) - 2*x*y*z**2*(1 - x)*(1 - y)**2*(1 - z)**2

    return np.vstack((first, second, third))

def p_ex(sd):
    x = sd.nodes[0, :]
    y = sd.nodes[1, :]
    z = sd.nodes[2, :]

    return x*y*z*(1 - x)*(1 - y)*(1 - z)

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
    st = Stokes3DHyb(keyword)
    spp, rhs, M, curl = st.matrix_and_rhs(mdg, source, bc_val)

    # solve the problem
    x = sps.linalg.spsolve(spp, rhs)
    q = x[:mdg.num_subdomain_faces()]
    p = x[-mdg.num_subdomain_cells():]

    # post process vorticity
    r = sps.linalg.spsolve(M, curl.T @ q)
    return r, q, p

def main(mdg, stokes, ne, keyword="flow"):
    # set the data
    bc_val = []
    source = []
    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        b_face_centers = sd.face_centers[:, b_faces]

        faces, _, sign = sps.find(sd.cell_faces)
        sign = sign[np.unique(faces, return_index=True)[1]]

        bc_val.append(np.zeros(sd.num_faces))
        source.append(vector_source(sd))

    r, q, p = stokes(mdg, keyword, np.hstack(source), np.hstack(bc_val))

    # post process vorticity
    proj_r = pg.eval_at_cell_centers(mdg, ne(keyword))
    P0r = (proj_r * r).reshape((3, -1), order="F")

    # post process velocity
    proj_q = pg.proj_faces_to_cells(mdg)
    P0q = (proj_q * q).reshape((3, -1), order="F")

    # compute the error
    return error.compute(sd, P0r, r_ex, P0q, q_ex, p, p_ex)

    #for sd, data in mdg.subdomains(return_data=True):
    #   data[pp.STATE] = {"P0r": P0r, "P0q": P0q, "p": p, "err_p": p_ex(sd) - p, "p_ex": p_ex(sd)}

    #save = pp.Exporter(mdg, "sol")
    #save.write_vtu(["P0r", "P0q", "p", "err_p", "p_ex"])

if __name__ == "__main__":

    N = np.arange(9, 14)
    print("stokes")
    err = np.array([main(create_grid(n), solve_stokes, pg.Nedelec0) for n in N])
    error.order(err)

    print("stokes hyb")
    err_hyb = np.array([main(create_grid(n), solve_stokes_hyb, pg.Nedelec1) for n in N])
    error.order(err_hyb)
