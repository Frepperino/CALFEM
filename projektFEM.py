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

########################################
#           PREPROCESSING              #
########################################

# Boundary markers
MARKER_T_INF=0
MARKER_T_OUT=1
MARKER_T_IN=2

###  MESH  ###
g = cfg.Geometry()
# ha med i rapporten motsolsplacering

# OUTER
# corner                    # POINTS
g.point([0.0, 0.0])         # 0       
g.point([300.0, 0.0])       # 1
g.point([400.0, 200.0])     # 2
g.point([0.0, 200.0])       # 3
# half-circle
g.point([0.0, 100.0])       # 4
g.point([0.0, 125.0])       # 5
g.point([25.0, 100.0])      # 6
g.point([0.0, 75.0])        # 7

                                                 # CURVES
g.spline([0, 1], marker=MARKER_T_INF)            # 0
g.spline([1, 2], marker=MARKER_T_INF)            # 1
g.spline([2, 3], marker=MARKER_T_INF)            # 2
g.spline([3, 5], marker=MARKER_T_INF)            # 3
g.circle([5, 4, 6], marker=MARKER_T_OUT)         # 4
g.circle([6, 4, 7], marker=MARKER_T_OUT)         # 5
g.spline([7, 0], marker=MARKER_T_INF)            # 6

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

hole1 = create_hole([87.5, 100.0], 25.0, 8, 7, MARKER_T_IN)
hole2 = create_hole([87.5 * 2.0, 100.0], 25.0, 13, 11, MARKER_T_OUT)
hole3 = create_hole([87.5 * 3.0, 100.0], 25.0, 18, 15, MARKER_T_IN)

g.surface([0, 1, 2, 3, 4, 5, 6], [hole1, hole2, hole3])

cfv.draw_geometry(g)

mesh = cfm.GmshMesh(g)
mesh.el_type = 2
mesh.dofs_per_node = 1
mesh.el_size_factor = 10

coords, edof, dofs, bdofs, elementmarkers = mesh.create()

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
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_INF, 293, 1)
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_OUT, 285, 1)
bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_IN, 277, 1)

### FEM SETUP

ndof = len(coords)
[ex, ey] = cfc.coordxtr(edof,coords,dofs,3)
K=np.zeros((ndof, ndof))
f=np.zeros((ndof, 1))



########################################
#             SOLVER                   #
########################################
for i in range(len(ex)):
    # isotropic???
    kei = cfc.flw2te(ex[i,:], ey[i,:], [1], 80*np.identity(2))
    cfc.assem(edof[i,:], K, kei)
a, r = cfc.solveq(K, f, bc, bc_value)


#print("a: "+ str(a))
cfv.figure()
cfv.draw_nodal_values(a, coords, edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type)
print("T_min = " + str(min(a)))
print("T_max = " + str(max(a)))
cfv.colorbar()
cfv.showAndWait()
#