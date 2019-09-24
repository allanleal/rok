import firedrake as fire
import numpy as np


class PressureFixBC(fire.DirichletBC):
    def __init__(self, V, point, val, subdomain, method="topological"):
        super().__init__(V, val, subdomain, method)
        sec = V.dm.getDefaultSection()
        dm = V.mesh()._plex

        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        if len(point) > 3:
            raise ValueError('Invalid dimension for a point.')
        if len(point) != dim:
            raise ValueError('The provided point has incompatible dimension for the mesh.')

        (vStart, vEnd) = dm.getDepthStratum(0)
        indices = []
        tol = 1e-8
        if dim == 1:
            for pt in range(vStart, vEnd):
                x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
                if point[0] - tol <= x[0] <= point[0] + tol:
                    if dm.getLabelValue("pyop2_ghost", pt) == -1:
                        indices = [pt]
                    break
        elif dim == 2:
            for pt in range(vStart, vEnd):
                x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
                if point[0] - tol <= x[0] <= point[0] + tol and point[1] - tol <= x[1] <= point[1] + tol:
                    if dm.getLabelValue("pyop2_ghost", pt) == -1:
                        indices = [pt]
                    break
        else:
            for pt in range(vStart, vEnd):
                x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
                if point[0] - tol <= x[0] <= point[0] + tol and point[1] - tol <= x[1] <= point[1] + tol \
                        and point[2] - tol <= x[2] <= point[2] + tol:
                    if dm.getLabelValue("pyop2_ghost", pt) == -1:
                        indices = [pt]
                    break

        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))

        self.nodes = np.asarray(nodes, dtype=np.int)

        print("Fixing nodes %s" % self.nodes)


def boundaryNameToIndex(name):
    if type(name) is int:
        return name
    return 1 + ['left', 'right', 'bottom', 'top', 'front', 'back'].index(name)


def vectorComponentNameToIndex(name):
    if type(name) is int:
        return name
    return ['x', 'y', 'z'].index(name)


class DirichletExpressionBC(fire.DirichletBC):
    def __init__(self, function_space, expr, boundary, method="topological"):
        self.uboundary = fire.Function(function_space)
        self.interpolator = fire.Interpolator(expr, self.uboundary)
        super().__init__(function_space, self.uboundary, boundary, method)


    def update(self):
        self.interpolator.interpolate()
