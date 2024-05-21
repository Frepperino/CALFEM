import numpy as np
import calfem.core as cfc
import geom2

###################################### Preprocessor
A = 10
k = 5
L = 6

coord, edof, dofs, bdofs, bc, bc_value, element_markers = geom2.generate_mesh(False)

ndof = len(coord)
[ex, ey] = cfc.coordxtr(edof,coord,dofs,3)
K=np.zeros((ndof, ndof))
F=np.zeros((ndof, 1))



######################################### SOLVER
for i in range(len(ex)):
    kei = cfc.flw2te(ex[i,:], ey[i,:], [1], np.identity(2))
    cfc.assem(edof[i,:], K, kei)
print(K)
a, r = cfc.solveq(K, F, bc, bc_value)

print("a: "+ str(a))