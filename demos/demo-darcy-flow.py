from firedrake import *
import numpy as np

rho = Constant(1000.0)  # water density (in units of kg/m3)
mu = Constant(8.9e-4)  # water viscosity (in units of Pa*s)
k = Constant(1e-12)  # rock permeability (in units of m2)

pL = Constant(100.0e5)  # left pressure is 100 bar === 100e5 Pa
pR = Constant(1.0e5)  # right pressure is 1 bar === 1e5 Pa

mesh = UnitSquareMesh(32, 32, quadrilateral=True)

BDM = FunctionSpace(mesh, "RTCF", 1)
DG = FunctionSpace(mesh, "DG", 0)
W = BDM * DG

n = FacetNormal(mesh)

u, p = TrialFunctions(W)
w, v = TestFunctions(W)
f = Constant(0.0)

k = Function(DG)

maxkorder = 14
minkorder = 11
k.dat.data[:] = 10 ** np.random.randint(-maxkorder, -minkorder, size=len(k.dat.data)).astype(float)

kappa = k / mu

a = (dot(u, w) - p * div(kappa * w) - v * div(rho * u)) * dx
L = f * v * dx - dot(kappa * w * pL, n) * ds(1) - dot(kappa * w * pR, n) * ds(2)

w = Function(W)

parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solve(a == L, w, solver_parameters=parameters)

u, p = w.split()

u.rename("u")
p.rename("p")

File("darcy-u-p-equation.pvd").write(u, p)
