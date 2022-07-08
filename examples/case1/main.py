import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import matplotlib.pyplot as plt


def main():
    keyword = "stokes"

    #dim = 3
    #nodes = np.array([[0, 1, 0, 0],
    #                  [0, 0, 1, 0],
    #                  [0, 0, 0, 1]])

    #indices = [0, 2, 1, 1, 2, 3, 2, 0, 3, 3, 0, 1]
    #indptr = [0, 3, 6, 9, 12]
    #face_nodes = sps.csc_matrix(([True] * 12, indices, indptr))
    #cell_faces = sps.csc_matrix([[1], [1], [1], [1]])
    #name = "test"

    #g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)

    g = pp.StructuredTetrahedralGrid([5]*3, [1]*3)
    pg.convert_from_pp(g)
    g.compute_geometry()

    parameters = {"second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells))}
    data = {pp.PARAMETERS: {keyword: parameters}}
    data[pp.DISCRETIZATION_MATRICES] = {keyword: {}}
    nd1 = pg.Nedelec1(keyword)

    nd1.discretize(g, data)

    M = data[pp.DISCRETIZATION_MATRICES][keyword]["mass"]
    curl = data[pp.DISCRETIZATION_MATRICES][keyword]["curl"]

    A = curl * sps.linalg.spsolve(M, curl.T)
    div = pg.div(g)
    spp = sps.bmat([[A, -div.T], [div, None]])

    rt0 = pp.RT0(keyword)
    rt0.discretize(g, data)

    proj = data[pp.DISCRETIZATION_MATRICES][keyword]["vector_proj"]
    rhs = np.zeros(spp.shape[0])
    rhs[:g.num_faces] = proj.T * np.ones(3*g.num_cells)

    x = sps.linalg.spsolve(spp, rhs)
    u = x[:g.num_faces]
    p = x[-g.num_cells:]

    P0u = rt0.project_flux(g, u, data)

    save = pp.Exporter(g, "sol")
    save.write_vtu([("P0u", P0u), ("p", p)])

    #spp3 = sps.bmat([[M, -curl.T, None], [curl, None, -div.T], [None, div, None]])
    #x = sps.linalg.spsolve(spp3, np.zeros(spp3.shape[0]))


    plt.spy(spp3)
    plt.show()

    import pdb; pdb.set_trace()



if __name__ == "__main__":
    main()
