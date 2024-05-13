import numpy as np
import calfem.core as cfc

###################################### Preprocessor
A = 10
k = 5
L = 6

#coord = np.array([[2], [4], [6], [8]])
edof = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])

nen = 2
ndof = 4
nelm = 3
dof = np.array([1, 2, 3, 4])

#[ex, ey] = cfc.coordxtr(edof,coord,dof,nen)
K=np.zeros((ndof, ndof))
F=np.array([100*2/2, 100 *2, 100*2, 100*2 + 15*A])

F[3] = 15

bc= np.array([
    1
])
bc_val = np.array([0])



######################################### SOLVER
for i in range(nelm):
    fei = [100, 100]
    kei = cfc.spring1e(k*10/2)
    cfc.assem(edof[i,:], K, kei)
    bc[i]

print(K)

a, r = cfc.solveq(K, F, bc, bc_val)

print("a: "+ str(a))