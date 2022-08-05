import numpy as np
import scipy.sparse as sps
import porepy as pp
import pygeon as pg

class Stokes2DHyb:
    def __init__(self, keyword):
        self.keyword = keyword

    def matrix_and_rhs(self, mdg, source, bc_val):
        # discretize
        M = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, None)

        face_mass = pg.face_mass(mdg)
        curl = face_mass * pg.curl(mdg)
        div = pg.div(mdg)

        # assemble the saddle point problem
        A = curl * sps.linalg.spsolve(M.tocsc(), curl.T.tocsc())
        spp = sps.bmat([[A, -div.T], [div, None]], format="csc")

        # assemble the right-hand side
        rhs = np.zeros(spp.shape[0])
        rhs[:face_mass.shape[0]] = face_mass * source + bc_val

        return spp, rhs, M, curl


class Stokes3DHyb:
    def __init__(self, keyword):
        self.keyword = keyword

    def matrix_and_rhs(self, mdg, source, bc_val):
        # discretize
        discr_ne1 = pg.Nedelec1(self.keyword)
        M = pg.numerics.innerproducts.lumped_mass_matrix(mdg, 2, discr_ne1)

        curl_ne = sps.bmat([[discr_ne1.assemble_curl(sd) for sd in mdg.subdomains()]])
        face_mass = pg.face_mass(mdg)
        curl = face_mass * curl_ne
        div = pg.div(mdg)

        # assemble the saddle point problem
        A = curl * sps.linalg.spsolve(M.tocsc(), curl.T.tocsc())
        spp = sps.bmat([[A, -div.T], [div, None]], format="csc")

        # assemble the right-hand side
        rhs = np.zeros(spp.shape[0])
        rhs[:face_mass.shape[0]] = face_mass * source + bc_val

        return spp, rhs, M, curl

class Stokes:
    def __init__(self, keyword):
        self.keyword = keyword

    def matrix_and_rhs(self, mdg, source, bc_val):
        # discretize
        M = pg.ridge_mass(mdg)

        face_mass = pg.face_mass(mdg)
        curl = face_mass * pg.curl(mdg)
        div = pg.div(mdg)

        # assemble the saddle point problem
        spp = sps.bmat([[M, -curl.T, None], [curl, None, -div.T], [None, div, None]], format="csc")

        # assemble the right-hand side
        rhs = np.zeros(spp.shape[0])
        rhs[M.shape[0]:M.shape[0]+face_mass.shape[0]] = face_mass * source + bc_val

        return spp, rhs
