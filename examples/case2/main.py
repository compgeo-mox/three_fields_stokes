import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
from stokes import Stokes

def source(g):
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
    first = -4*(x**4*(6*y-3)+x**3*(6-12*y)+3*x**2*(4*y**3-6*y**2+4*y-1)-6*x*y*(2*y**2-3*y+1)+y*(2*y**2-3*y+1))
    second = 4*(2*x**3*(6*y**2-6*y+1)-3*x**2*(6*y**2-6*y+1)+x*(6*y**4-12*y**3+12*y**2-6*y+1)-3*(y-1)**2*y**2)
    return np.vstack((-g.cell_volumes*first, -g.cell_volumes*second, np.zeros(g.num_cells))).ravel(order="F")

def p_ex(g):
    return np.zeros(g.num_cells)

def u_ex(g):
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
    first = -2*x*y*(x-1)*(y-1)*x*(x-1)*(2*y-1)
    second = 2*x*y*(x-1)*(y-1)*y*(y-1)*(2*x-1)
    return np.vstack((first, second, np.zeros(g.num_cells)))

def main(n):
    keyword = "flow" #"stokes"

    sd = pp.StructuredTriangleGrid([n]*2, [1]*2)
    mdg = pp.meshing.subdomains_to_mdg([[sd]])
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    for sd, data in mdg.subdomains(return_data=True):
        parameters = {"second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))}
        data[pp.PARAMETERS] = {keyword: parameters}
        data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}


    mass_rt0 = pg.face_mass(mdg)
    M = pg.lumped_mass_matrix(mdg, 2) <================

    curl = mass_rt0 * pg.curl(mdg)
    div = pg.div(mdg)

    import pdb; pdb.set_trace()
    A = curl * sps.linalg.spsolve(M.tocsc(), curl.T)
    spp = sps.bmat([[A, -div.T], [div, None]], format="csc")

    proj = data[pp.DISCRETIZATION_MATRICES][keyword]["vector_proj"]
    rhs = np.zeros(spp.shape[0])
    rhs[:g.num_faces] = proj.T * source(g)

    x = sps.linalg.spsolve(spp, rhs)
    u = x[:g.num_faces]
    p = x[-g.num_cells:]

    P0u = rt0.project_flux(g, u, data)

    err_u = np.sqrt(np.trace((u_ex(g) - P0u) @ sps.diags(g.cell_volumes) @ (u_ex(g) - P0u).T))
    err_p = np.sqrt((p_ex(g) - p) @ sps.diags(g.cell_volumes) @ (p_ex(g) - p).T)

    print(np.mean(g.cell_diameters()), err_u, err_p)

    #save = pp.Exporter(g, "sol")
    #save.write_vtu([("P0u", P0u), ("p", p), ("u_ex", u_ex(g))])

if __name__ == "__main__":
    N = 2**np.arange(3, 8)
    [main(n) for n in N]
