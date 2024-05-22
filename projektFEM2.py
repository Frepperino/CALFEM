import math
from scipy.integrate import solve_ivp
from plantml import plantml

import calfem.core as cfc
import numpy as np
import calfem.utils as cfu

# Mesh
import calfem.geometry as cfg
import calfem.mesh as cfm

# Visual
import calfem.vis_mpl as cfv
import matplotlib as mpl
mpl.use('TkAgg')


alpha_n = 40
alpha_c = 120
T_inf = 293
T_out = 285
T_in = 277

thickness = 1.6
density = 540


########################################
#           PREPROCESSING              #
########################################

# Boundary markers
MARKER_T_CONVECTION=5
MARKER_T_OUT=1
MARKER_T_IN=2
MARKER_U_DISPLACEMENT=6
MARKER_U_DISPLACEMENT_X=9

###  MESH  ###
g = cfg.Geometry()
# ha med i rapporten motsolsplacering

# OUTER
# corner                    # POINTS
g.point([0.0, 0.0])         # 0       
g.point([0.300, 0.0])       # 1
g.point([0.400, 0.200])     # 2
g.point([0.0, 0.200])       # 3
# half-circle
g.point([0.0, 0.100])       # 4
g.point([0.0, 0.125])       # 5
g.point([0.025, 0.100])      # 6
g.point([0.0, 0.075])        # 7

                                                 # CURVES
g.spline([0, 1], marker=MARKER_U_DISPLACEMENT)            # 0
g.spline([1, 2], marker=MARKER_U_DISPLACEMENT)            # 1
g.spline([2, 3], marker=MARKER_T_CONVECTION)            # 2
g.spline([3, 5], marker=MARKER_U_DISPLACEMENT_X)            # 3
g.circle([5, 4, 6], marker=MARKER_T_OUT)         # 4
g.circle([6, 4, 7], marker=MARKER_T_OUT)         # 5
g.spline([7, 0], marker=MARKER_U_DISPLACEMENT_X)            # 6

# HOLES
def create_hole(center, radius, point_startid, curve_startid, marker):
    g.point(center)
    g.point([center[0], center[1] + radius])
    g.point([center[0] + radius, center[1]])
    g.point([center[0], center[1] - radius])
    g.point([center[0] - radius, center[1]])
    g.circle([point_startid + 1, point_startid, point_startid + 2], marker=marker)
    g.circle([point_startid + 2, point_startid, point_startid + 3], marker=marker)
    g.circle([point_startid + 3, point_startid, point_startid + 4], marker=marker)
    g.circle([point_startid + 4, point_startid, point_startid + 1], marker=marker)
    return [curve_startid, curve_startid + 1, curve_startid + 2, curve_startid + 3]

hole1 = create_hole([0.0875, 0.100], 0.025, 8, 7, MARKER_T_IN)
hole2 = create_hole([0.0875 * 2.0, 0.100], 0.025, 13, 11, MARKER_T_OUT)
hole3 = create_hole([0.0875 * 3.0, 0.100], 0.025, 18, 15, MARKER_T_IN)

g.surface([0, 1, 2, 3, 4, 5, 6], [hole1, hole2, hole3])

cfv.draw_geometry(g)

mesh = cfm.GmshMesh(g)
mesh.el_type = 2
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.023
mesh.returnBoundaryElements = True

coords, edof, dofs, bdofs, elementmarkers, boundaryElements = mesh.create()

cfv.figure()

# Draw the mesh.
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofsPerNode,
    el_type=mesh.elType,
    filled=True,
    title="Battery"
        )

# Boundary Conditions
bc, bc_value = np.array([], 'i'), np.array([], 'f')
#bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_INF, 293, 1)
#bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_OUT, 285, 1)
#bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_IN, 277, 1)

### FEM SETUP

ndof = len(coords)
nelem = len(edof)
[ex, ey] = cfc.coordxtr(edof,coords,dofs,3)
K=np.zeros((ndof, ndof))
f=np.zeros((ndof, 1))



########################################
#             SOLVER                   #
########################################

def convection_boundary(boundaryElements, K, f, T, alpha, marker):
    for bel in boundaryElements[marker]:
        node1 = bel['node-number-list'][0]
        node2 = bel['node-number-list'][1]
        node1i = node1 - 1
        node2i = node2 - 1

        Kce = np.array([[2, 1], [1, 2]])
        L = math.sqrt((coords[node1i][0] - coords[node2i][0])**2 + (coords[node1i][1]-coords[node2i][1])**2)
        
        # thickness här?

        Kce= (alpha*L/6)*Kce*thickness
        fbc = np.array([[1/2], [1/2]])
        fbc*=alpha*T*L*thickness

        cfc.assem(np.array([node1, node2]), K, Kce, f, fbc)

for i in range(nelem):
    kei = cfc.flw2te(ex[i,:], ey[i,:], [thickness], 80*np.eye(2))
    cfc.assem(edof[i,:], K, kei)

convection_boundary(boundaryElements, K, f, T_inf, alpha_n, MARKER_T_CONVECTION)
convection_boundary(boundaryElements, K, f, T_out, alpha_c, MARKER_T_OUT)
convection_boundary(boundaryElements, K, f, T_in, alpha_c, MARKER_T_IN)

a, r = cfc.solveq(K, f, bc, bc_value)

#print("a: "+ str(a))
cfv.figure()
cfv.draw_nodal_values(a, coords, edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type)
print("T_min = " + str(min(a)))
print("T_max = " + str(max(a)))
cfv.colorbar()



########################################
#             TRANSIENT                #
########################################

#Ce eller Ci inte båda looool

#Define the effect functions
def f_1(t):
    return 100*np.e**(-144*((600 - t)/3600)**2) * 1000

def f_2(t):
    return 88.42*1000 if t <= 600 else 0

K=np.zeros((ndof, ndof))
f_l=np.zeros((ndof, 1))
f_conv = np.zeros((ndof, 1))
C=np.zeros((ndof, ndof))



ts = np.linspace(0, 3600, 100)
y0 = np.full((ndof,), 293)

ys = np.empty((ts.size, y0.size))
ys[0, :] = y0
#print(ys)
# t = t0
# h = (tf - t0) / N
# implA = kg.inv(np.eye(len(y0)) - h*A)
for i in range(nelem):
    kei, fli = cfc.flw2te(ex[i,:], ey[i,:], [thickness], 80*np.eye(2), 1)
    A = np.linalg.det(np.vstack((np.ones((3, )), ex[i,:], ey[i,:])))/2
    Cei = plantml(ex[i,:],ey[i,:], density*3600*thickness)
    cfc.assem(edof[i,:], K, kei, f_l, fli)
    cfc.assem(edof[i,:], C, Cei)

convection_boundary(boundaryElements, K, f_conv, T_inf, alpha_n, MARKER_T_CONVECTION)
convection_boundary(boundaryElements, K, f_conv, T_out, alpha_c, MARKER_T_OUT)
convection_boundary(boundaryElements, K, f_conv, T_in, alpha_c, MARKER_T_IN)
# C_inv = np.linalg.inv(C)
# CK = -1*C_inv@K
# CFB = C_inv@f_conv.flatten()
# CFl = C_inv@f_l.flatten()

# def IVPTHANG(t,a):
#     return CK @ a + CFB + CFl*f_1(t)

# solution = solve_ivp(IVPTHANG, (0, 3600), y0, method="Radau", t_eval=ts)

dt = ts[1] - ts[0]
for t_i in range(1, len(ts)):
    f=f_l*f_1(ts[t_i]) + f_conv
    temp1 = dt*f.flatten() + (C@(np.transpose(ys[t_i - 1])))
    temp2 = (C + dt*K)
    #print(temp1)
    #print("f: " + str((temp1).shape))
    # print("temp1: " + str(temp1.shape))
    # print("Cinv: " + str(Cinv.shape))
    # print("Cinv * temp1: " + str((Cinv @ temp1).shape))
    ys[t_i] = np.transpose(np.linalg.inv(temp2)@(temp1))






plot_ts = [0, 15, 30, 45, 60, 80]
max_ys = np.zeros((100,))
min_ys = np.zeros((100,))
deviation = np.zeros((100,))

for i in range(len(ts)):
    max_val = max(ys[i])
    min_val = min(ys[i])
    max_dev = max(abs(max_val - T_inf), abs(min_val - T_inf))
    max_ys[i] = max_val
    min_ys[i] = min_val
    deviation[i] = max_dev



for i in range(len(plot_ts)):
    cfv.figure()
    cfv.draw_nodal_values(np.transpose(ys[plot_ts[i]]), coords, edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type)
    print("T_min = " + str(min(ys[plot_ts[i]])))
    print("T_max = " + str(max(ys[plot_ts[i]])))
    cfv.colorbar()
    cfv.title("Time = " + str(ts[plot_ts[i]]))

cfv.figure()
cfv.plt.plot(ts, max_ys)
cfv.title("Max Ys")
cfv.figure()
cfv.plt.plot(ts, min_ys)
cfv.title("Min Ys")
cfv.figure()
cfv.plt.plot(ts, deviation)
cfv.title("Deviation")



###################### VON MISES ######################

vonmises_temp = ys[np.argmax(deviation)]

ep = [2, thickness]
D = cfc.hooke(2, 5e9, 0.36)                                 ## 1 here????
nu = 0.36
expansion = 60e-6
E = 5e9
D = E/(1-2*nu)/(1+nu)*np.array([[1-nu, nu, 0], [nu, 1-nu, 0], [0, 0, 1/2*(1-2*nu)]])
# D = np.insert(D, 2, values = [0, 0, 0], axis = 0)
# D = np.insert(D, 2, values = [0, 0, D[0,0], 0], axis = 1)
print("D: " + str(D))
cfv.plt

mesh = cfm.GmshMesh(g)
mesh.el_type = 2
mesh.dofs_per_node = 2
mesh.el_size_factor = 0.023
mesh.returnBoundaryElements = True

coords, edof, dofs, bdofs, elementmarkers, boundaryElements = mesh.create()

n_dofs = np.size(dofs)
print(n_dofs)
print(ndof)
ex, ey = cfc.coordxtr(edof, coords, dofs)
K = np.zeros((n_dofs,n_dofs))
f = np.zeros((n_dofs,1))



for eltopo, elx, ely in zip(edof, ex, ey):
    Ke = cfc.plante(elx, ely, ep, D)
    dT = sum([vonmises_temp[(eltopo[0] - 1)//2], vonmises_temp[(eltopo[1] - 1)//2], vonmises_temp[(eltopo[2]-1)//2]]) / 3 - T_inf
    eps_dT = (1+nu)*np.transpose(np.array([1, 1, 0]))*expansion*dT
    #print((D@eps_dT).reshape(3,1).T.shape)
    #Deps = D@eps_dT.flatten()
    #rint(np.array([(D@eps_dT.flatten())[0, i] for i in range(Deps.shape[1])]))
    #for e in range(4):
    #    print((D@eps_dT.flatten())[0, e])
    #print(str((D@eps_dT.flatten()).shape) + " " + str((D@eps_dT.flatten())[0].shape))
    fe = cfc.plantf(elx, ely, ep, (D@eps_dT).reshape(3,1).T)
    # Deps^dT
    # D@alphadT

    cfc.assem(eltopo, K, Ke, f, fe)


bc = np.array([],'i')
bcVal = np.array([],'f')

# FIXA X BARA DISPLACEMENET AOPWDPOI 
bc, bcVal = cfu.applybc(bdofs, bc, bcVal, MARKER_U_DISPLACEMENT, 0.0, 0)
bc, bcVal = cfu.applybc(bdofs, bc, bcVal, MARKER_U_DISPLACEMENT_X, 0.0, 1)


#cfu.applyforce(bdofs, f, 7, 10e5, 1)
a,r = cfc.solveq(K,f,bc,bcVal)

ed = cfc.extract_eldisp(edof,a)

von_mises = []

# For each element:
for eltopo, elx, ely, eld in zip(edof, ex, ey, ed):
    # Determine element stresses and strains in the element.
    es, et = cfc.plants(elx, ely, ep, D, eld)
    dT = sum([vonmises_temp[(eltopo[0] - 1)//2], vonmises_temp[(eltopo[1]-1)//2], vonmises_temp[(eltopo[2]-1)//2]]) / 3 - T_inf
    #print("es: " + str(es))
    # Calc and append effective stress to list
    #print((np.array([1, 1, 0]).T*expansion*dT).shape)
    #print((et.T.flatten() - np.array([1, 1, 0]).T*expansion*dT).shape)
    es = (D@(et.T.flatten() - (1+nu)*np.array([1, 1, 0]).T*expansion*dT)).flatten()
    sigz = nu*(es[0] + es[1]) - expansion*E*dT
    
    #print(es.shape)
    von_mises.append(np.sqrt(np.power(es[0],2) + np.power(es[1],2) + np.power(sigz,2) - es[0]*es[1] - es[0]*sigz - es[1]*sigz + 3*np.power(es[2],2)))

    ## es: [sigx sigy tauxy]

cfv.figure(fig_size=(10,10))
cfv.draw_element_values(von_mises, coords, edof, mesh.dofs_per_node, mesh.el_type, None, draw_elements=False, draw_undisplaced_mesh=False, title="Example 6 - Effective stress")
cfv.colorbar()

cfv.figure(fig_size=(10,10))
cfv.draw_displacements(a * 100, coords, edof, mesh.dofs_per_node, mesh.el_type, draw_undisplaced_mesh=True, title="Example 06 - Displacements")

cfv.showAndWait()
print(max(von_mises))




# p-type
# när har vi 3x3 typ