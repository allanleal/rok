def boundaryNameToIndex(name):
    if type(name) is int:
        return name
    return 1 + ['left', 'right', 'bottom', 'top', 'front', 'back'].index(name)


def vectorComponentNameToIndex(name):
    if type(name) is int:
        return name
    return ['x', 'y', 'z'].index(name)

