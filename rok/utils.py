import firedrake as fire


def boundaryNameToIndex(name):
    if type(name) is int:
        return name
    return 1 + ["left", "right", "bottom", "top", "front", "back"].index(name)


def vectorComponentNameToIndex(name):
    if type(name) is int:
        return name
    return ["x", "y", "z"].index(name)


class DirichletExpressionBC(fire.DirichletBC):
    def __init__(self, function_space, expr, boundary, method="topological"):
        self.uboundary = fire.Function(function_space)
        self.interpolator = fire.Interpolator(expr, self.uboundary)
        super().__init__(function_space, self.uboundary, boundary, method)

    def update(self):
        self.interpolator.interpolate()
