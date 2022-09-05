import numpy as np
import scipy.sparse as sps

import pygeon as pg


def compute(sd, r, r_ex, P0q, q_ex, p, p_ex):

    if sd.dim == 3:
        norm_r = np.sqrt(np.trace((r_ex(sd) @ sps.diags(sd.cell_volumes) @ r_ex(sd).T)))
        err_r = (
            np.sqrt(
                np.trace(
                    ((r_ex(sd) - r) @ sps.diags(sd.cell_volumes) @ (r_ex(sd) - r).T)
                )
            )
            / norm_r
        )
    else:
        mass = pg.Lagrange("flow").assemble_mass_matrix(sd, None)
        norm_r = np.sqrt(r_ex(sd) @ mass @ r_ex(sd).T)
        err_r = np.sqrt((r_ex(sd) - r) @ mass @ (r_ex(sd) - r).T) / norm_r

    norm_q = np.sqrt(np.trace(q_ex(sd) @ sps.diags(sd.cell_volumes) @ q_ex(sd).T))
    err_q = (
        np.sqrt(
            np.trace((q_ex(sd) - P0q) @ sps.diags(sd.cell_volumes) @ (q_ex(sd) - P0q).T)
        )
        / norm_q
    )

    # norm_p = np.sqrt(p_ex(sd) @ sps.diags(sd.cell_volumes) @ p_ex(sd).T)
    # err_p = np.sqrt((p_ex(sd) - p) @ sps.diags(sd.cell_volumes) @ (p_ex(sd) - p).T) / norm_p

    norm_p = cell_error(sd, np.zeros(sd.num_cells), p_ex(sd))
    err_p = cell_error(sd, p, p_ex(sd)) / norm_p

    h = np.mean(sd.cell_diameters())
    return h, err_r, err_q, err_p, sd.num_cells, sd.num_faces, sd.num_ridges


def cell_error(sd, sol, sol_ex):
    cell_nodes = sd.cell_nodes()
    err = 0
    for c in np.arange(sd.num_cells):
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
        nodes_loc = cell_nodes.indices[loc]
        err_loc = sol_ex[nodes_loc] - sol[c]

        err += sd.cell_volumes[c] * err_loc @ err_loc.T / (sd.dim + 1)
    return np.sqrt(err)


def order(vals):
    r_order = np.log(vals[:-1, 1] / vals[1:, 1]) / np.log(vals[:-1, 0] / vals[1:, 0])
    q_order = np.log(vals[:-1, 2] / vals[1:, 2]) / np.log(vals[:-1, 0] / vals[1:, 0])
    p_order = np.log(vals[:-1, 3] / vals[1:, 3]) / np.log(vals[:-1, 0] / vals[1:, 0])

    print("dofs\n", vals[:, 4:].astype(int))

    print("mesh size\n", vals[:, 0])

    print("error\n", vals[:, 1])
    print("r_order\n", r_order)

    print("error\n", vals[:, 2])
    print("q_order\n", q_order)

    print("error\n", vals[:, 3])
    print("p_order\n", p_order)
