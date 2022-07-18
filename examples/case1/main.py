import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

import sys; sys.path.append("../../src/")
from stokes import Stokes

def main():
    keyword = "stokes"

    g = pp.StructuredTetrahedralGrid([5]*3, [1]*3)
    pg.convert_from_pp(g)
    g.compute_geometry()

    parameters = {"second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells))}
    data = {pp.PARAMETERS: {keyword: parameters}}

    st = Stokes(keyword, g)
    st.discretize(data)

    spp = st.assemble_hybridized(data)

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

if __name__ == "__main__":
    main()
