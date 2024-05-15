import math
from scipy.integrate import solve_ivp
from plantml import plantml

# Calc
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
g.spline([0, 1])            # 0
g.spline([1, 2])            # 1
g.spline([2, 3], marker=MARKER_T_CONVECTION)            # 2
g.spline([3, 5])            # 3
g.circle([5, 4, 6], marker=MARKER_T_OUT)         # 4
g.circle([6, 4, 7], marker=MARKER_T_OUT)         # 5
g.spline([7, 0])            # 6

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
mesh.el_size_factor = 0.015
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
        
        Kce= (alpha*L/6)*Kce
        fbc = np.array([[1/2], [1/2]])
        fbc*=alpha*T*L

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




def eulerint(a0, ts):
    ys = np.empty((ts.size, a0.size))
    ys[0, :] = a0
    print(ys)
    # t = t0
    # h = (tf - t0) / N
    # implA = kg.inv(np.eye(len(y0)) - h*A)
    for i in range(nelem):
        kei, fli = cfc.flw2te(ex[i,:], ey[i,:], [1], 80*np.eye(2), 1)
        Cei = plantml(ex[i,:],ey[i,:], density*3600*thickness)
        cfc.assem(edof[i,:], K, kei, f_l, fli)
        cfc.assem(edof[i,:], C, Cei)
    
    convection_boundary(boundaryElements, K, f_conv, T_inf, alpha_n, MARKER_T_CONVECTION)
    convection_boundary(boundaryElements, K, f_conv, T_out, alpha_c, MARKER_T_OUT)
    convection_boundary(boundaryElements, K, f_conv, T_in, alpha_c, MARKER_T_IN)

    dt = ts[1] - ts[0]
    for t_i in range(1, len(ts)):
        f=f_l*f_1(ts[t_i]) + f_conv
        temp1 = dt*f.flatten() + (C@(np.transpose(ys[t_i - 1])))
        temp2 = (C + dt*K)
        #print("f: " + str((temp1).shape))
        # print("temp1: " + str(temp1.shape))
        # print("Cinv: " + str(Cinv.shape))
        # print("Cinv * temp1: " + str((Cinv @ temp1).shape))
        ys[t_i] = np.linalg.inv(temp2)@(temp1)
    return ys

ts = np.linspace(0, 3600, 25)
y0 = np.full((1, ndof), 273)
ys = eulerint(y0, ts)

for i in range(0, 4):
    t = len(ts)//4
    cfv.figure()
    cfv.draw_nodal_values(np.transpose(ys[t]), coords, edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type)
    print("T_min = " + str(min(a)))
    print("T_max = " + str(max(a)))
    cfv.colorbar()



cfv.showAndWait()
#