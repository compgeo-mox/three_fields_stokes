import scipy.sparse as sps
import porepy as pp
import pygeon as pg


class Stokes:
    def __init__(self, keyword, g):
        self.keyword = keyword
        self.g = g
        self.nd1 = pg.Nedelec1(keyword)
        self.rt0 = pp.RT0(keyword)

    def discretize(self, data):
        data[pp.DISCRETIZATION_MATRICES] = {self.keyword: {}}

        self.nd1.discretize(self.g, data)
        self.M = data[pp.DISCRETIZATION_MATRICES][self.keyword]["mass"]

        self.rt0.discretize(self.g, data)
        mass_rt0 = data[pp.DISCRETIZATION_MATRICES][self.keyword]["mass"]

        self.curl = mass_rt0 * data[pp.DISCRETIZATION_MATRICES][self.keyword]["curl"]
        self.div = pg.div(self.g)

    def assemble_hybridized(self, data):
        A = self.curl * sps.linalg.spsolve(self.M.tocsc(), self.curl.T)
        return sps.bmat([[A, -self.div.T], [self.div, None]], format="csc")

    def assemble(self):
        return sps.bmat(
            [
                [self.M, -self.curl.T, None],
                [self.curl, None, -self.div.T],
                [None, self.div, None],
            ],
            format="csc",
        )
