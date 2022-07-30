import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg


def source(g):
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
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
        (-g.cell_volumes * first, -g.cell_volumes * second, np.zeros(g.num_cells))
    ).ravel(order="F")


def p_ex(g):
    return np.zeros(g.num_cells)


def u_ex(g):
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
    first = -2 * x * y * (x - 1) * (y - 1) * x * (x - 1) * (2 * y - 1)
    second = 2 * x * y * (x - 1) * (y - 1) * y * (y - 1) * (2 * x - 1)
    return np.vstack((first, second, np.zeros(g.num_cells)))


def main(n):
    keyword = "flow"  # "stokes"

    sd = pp.StructuredTriangleGrid([n] * 2, [1] * 2)
    mdg = pp.meshing.subdomains_to_mdg([[sd]])
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    for sd, data in mdg.subdomains(return_data=True):
        parameters = {
            "second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))
        }
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}

    mass_rt0 = pg.face_mass(mdg)
    M = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, None)

    curl = mass_rt0 * pg.curl(mdg)
    div = pg.div(mdg)

    A = curl * sps.linalg.spsolve(M.tocsc(), curl.T.tocsc())
    spp = sps.bmat([[A, -div.T], [div, None]], format="csc")

    proj = data[pp.DISCRETIZATION_MATRICES][keyword]["vector_proj"]
    rhs = sps.lil_matrix((spp.shape[0], 1))
    rhs[: sd.num_faces] = proj.T * source(sd)

    x = sps.linalg.spsolve(spp, rhs.tocsc())
    u = x[: sd.num_faces]
    p = x[-sd.num_cells :]

    rt0 = pp.RT0(keyword)
    P0u = rt0.project_flux(sd, u, data)

    err_u = np.sqrt(
        np.trace((u_ex(sd) - P0u) @ sps.diags(sd.cell_volumes) @ (u_ex(sd) - P0u).T)
    )
    err_p = np.sqrt((p_ex(sd) - p) @ sps.diags(sd.cell_volumes) @ (p_ex(sd) - p).T)

    print(np.mean(sd.cell_diameters()), err_u, err_p)

    # save = pp.Exporter(g, "sol")
    # save.write_vtu([("P0u", P0u), ("p", p), ("u_ex", u_ex(g))])


if __name__ == "__main__":
    N = 2 ** np.arange(3, 8)
    [main(n) for n in N]
