import numpy as np
import calfem.core as cfc
import geom2

###################################### Preprocessor
A = 10
k = 5
L = 6

coord, edof, dofs, bdofs, bc, bc_value, element_markers = geom2.generate_mesh()

ndof = len(dofs) * 3
[ex, ey] = cfc.coordxtr(edof,coord,dofs,3)
K=np.zeros((ndof, ndof))
F=np.zeors(ndof)pp



######################################### SOLVER
for i in range(nelm):
    cfc.flw2te()
    kei = cfc.spring1e(k*10/2)
    cfc.assem(edof[i,:], K, kei)
    bc[i]

print(K)

a, r = cfc.solveq(K, F, bc, bc_val)

print("a: "+ str(a))