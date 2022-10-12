import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys

sys.path.append("../../src/")
from stokes import Stokes

def create_grid(n):
    # make the grid
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.FractureNetwork2d(domain=domain)

    mesh_size = 1/n
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size}

    mdg = network.mesh(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return mdg

def solve_stokes(mdg, keyword, source, bc_val_ridge, bc_val_face, bc_ess):
    # create the Stokes solver
    st = Stokes(keyword)
    spp, rhs = st.matrix_and_rhs(mdg, source, bc_val_face, bc_val_ridge)

    # solve the problem
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(bc_ess, np.zeros(bc_ess.size))
    x = ls.solve()

    r = x[:mdg.num_subdomain_ridges()]
    q = x[mdg.num_subdomain_ridges():mdg.num_subdomain_ridges()+mdg.num_subdomain_faces()]
    p = x[-mdg.num_subdomain_cells():]

    return r, q, p

def main(mdg, keyword="flow"):
    # set the data
    bc_val_face, bc_val_ridge, bc_ess, source = [], [], [], []
    for sd, data in mdg.subdomains(return_data=True):

        b_faces = sd.tags["domain_boundary_faces"]
        ridge_c = sd.nodes

        ess_faces = b_faces

        ess_ridges = np.zeros(sd.num_ridges, dtype=bool)
        top_ridges = np.logical_and.reduce((ridge_c[1, :] == 1, ridge_c[0, :] != 0, ridge_c[0, :] != 1))

        ess_cells = np.zeros(sd.num_cells, dtype=bool)
        ess_cells[0] = True

        bc_faces = np.zeros(sd.num_faces)
        bc_ridges = np.zeros(sd.num_ridges)
        bc_ridges[top_ridges] = 1

        bc_val_ridge.append(bc_ridges)
        bc_val_face.append(bc_faces)
        bc_ess.append(np.hstack((ess_ridges, ess_faces, ess_cells)))

        source.append(np.zeros(sd.num_faces))

        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells)),
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

    source = np.hstack(source)
    bc_val_ridge = np.hstack(bc_val_ridge)
    bc_val_face = np.hstack(bc_val_face)
    bc_ess = np.hstack(bc_ess)

    # Stokes solver
    r, q, p = solve_stokes(mdg, keyword,source, bc_val_ridge, bc_val_face, bc_ess)

    ridge_proj = pg.eval_at_cell_centers(mdg, pg.Lagrange1(keyword))
    face_proj = pg.eval_at_cell_centers(mdg, pg.RT0(keyword))
    cell_proj = pg.eval_at_cell_centers(mdg, pg.PwConstants(keyword))

    # post process vorticity
    cell_r = ridge_proj * r

    # post process velocity
    cell_q = (face_proj * q).reshape((3, -1), order="F")

    # post process pressure
    cell_p = cell_proj * p

    for _, data in mdg.subdomains(return_data=True):
       data[pp.STATE] = {"cell_r": cell_r, "cell_q": cell_q, "cell_p": cell_p}

    save = pp.Exporter(mdg, "sol")
    save.write_vtu(["cell_r", "cell_q", "cell_p"])

if __name__ == "__main__":
    main(create_grid(40))
