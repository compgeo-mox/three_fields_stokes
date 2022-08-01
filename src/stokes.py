import scipy.sparse as sps
import porepy as pp
import pygeon as pg

class Stokes2D:
    def __init__(self, keyword):
        self.keyword = keyword

    def matrix_and_rhs(self, mdg, vector_source):
        # discretize
        mass_rt0 = pg.face_mass(mdg)
        M = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, None)

        curl = mass_rt0 * pg.curl(mdg)
        div = pg.div(mdg)

        # assemble the saddle point problem
        A = curl * sps.linalg.spsolve(M.tocsc(), curl.T)
        spp = sps.bmat([[A, -div.T], [div, None]], format="csc")

        # assemble the right-hand side
        proj = pg.proj_cells_to_faces(mdg)
        rhs = sps.lil_matrix((spp.shape[0], 1))
        rhs[:proj.shape[0]] = proj * vector_source

        return spp, rhs, proj

class Stokes3D:
    def __init__(self, keyword):
        self.keyword = keyword

    def matrix_and_rhs(self, mdg, vector_source):
        # discretize
        discr_ne1 = pg.Nedelec1(self.keyword)
        M = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, discr_ne1)

        curl_ne = sps.bmat([[discr_ne1.assemble_curl(sd) for sd in mdg.subdomains()]])
        curl = pg.face_mass(mdg) * curl_ne
        div = pg.div(mdg)

        # assemble the saddle point problem
        print("do the hybridization")
        A = curl * sps.linalg.spsolve(M.tocsc(), curl.T.tocsc())
        print("done")
        spp = sps.bmat([[A, -div.T], [div, None]], format="csc")

        # assemble the right-hand side
        print("project the shit")
        proj = pg.proj_cells_to_faces(mdg)
        print("done")
        rhs = sps.lil_matrix((spp.shape[0], 1))
        rhs[:proj.shape[0]] = proj * vector_source

        return spp, rhs, proj
