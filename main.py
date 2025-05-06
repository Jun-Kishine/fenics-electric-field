import numpy as np
import matplotlib.pyplot as plt
import pyvista
import gmsh
import dolfinx
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, dirichletbc, locate_dofs_geometrical,
                         form, LinearProblem)
from dolfinx.mesh import create_rectangle, CellType
from ufl import TestFunction, TrialFunction, dx, grad, inner
from mpi4py import MPI

# === パラメータ設定 ===
charge_strength = 1.0     # 点電荷の強さ
charge_pos = np.array([0.0, 0.0])  # 点電荷の位置
domain_x = (-5.0, 5.0)
domain_y = (-4.0, 4.0)

# === Gmshでジオメトリ作成（導体を除いた領域） ===
gmsh.initialize()
gmsh.model.add("charge_domain")

# 外枠（計算領域）
outer = gmsh.model.occ.addRectangle(domain_x[0], domain_y[0], 0, 
                                    domain_x[1] - domain_x[0], domain_y[1] - domain_y[0])

# 導体（正方形）
square = gmsh.model.occ.addRectangle(2, 1, 0, 2, 2)

# 導体（三角形）
triangle = gmsh.model.occ.addPolygon([[-4, -2, 0], [-2, -2, 0], [-3, 0, 0]])

# 導体を除いた領域を定義
main_region = gmsh.model.occ.cut([(2, outer)], [(2, square), (2, triangle)])
gmsh.model.occ.synchronize()

# メッシュ生成
gmsh.model.mesh.generate(2)

# dolfinx用に読み込み
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

# === 機能空間と境界条件 ===
V = FunctionSpace(mesh, ("CG", 2))

# 導体境界上に φ = 0 の Dirichlet 条件を課す
def conductor_boundary(x):
    # gmshで除いた穴（導体）の周囲を捕まえる
    return np.logical_or(
        np.logical_and(2.0 <= x[0], x[0] <= 4.0, 1.0 <= x[1], x[1] <= 3.0),
        np.logical_and(
            -4 <= x[0], x[0] <= -2,
            np.abs(x[1] + 2*(x[0] + 3)) <= 0.5   # 三角形に収まるようにざっくり近似
        )
    )

bc = dirichletbc(
    dolfinx.fem.Function(V),
    locate_dofs_geometrical(V, conductor_boundary)
)

# === 弱形式（Poisson方程式） ===
u = TrialFunction(V)
v = TestFunction(V)

# 点電荷の近似（ガウス型分布）
delta_eps = 0.05
x = dolfinx.fem.Expression(
    charge_strength * 1/(np.pi*delta_eps**2) * 
    dolfinx.fem.exp(-((dolfinx.fem.SpatialCoordinate(mesh)[0] - charge_pos[0])**2 + 
                      (dolfinx.fem.SpatialCoordinate(mesh)[1] - charge_pos[1])**2) / delta_eps**2),
    V.element.interpolation_points()
)

a = inner(grad(u), grad(v)) * dx
L = x * v * dx

# === 数値解 ===
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg"})
uh = problem.solve()

# === 電場（ベクトル場）の計算 ===
V_vec = FunctionSpace(mesh, ("CG", 1), mesh.geometry.dim)
E = Function(V_vec)
E.interpolate(lambda x: -np.array(np.gradient(uh.x.array.reshape((V.dofmap.index_map.size_local,))), dtype=np.float64))

# === 可視化 ===
pyvista.set_jupyter_backend("none")
plotter = pyvista.Plotter()
grid = pyvista.plotting.wrap(mesh)
grid["phi"] = uh.x.array
warped = grid.warp_by_scalar("phi", factor=0.1)

plotter.add_mesh(warped, show_scalar_bar=True, cmap="coolwarm")
plotter.view_xy()
plotter.show(screenshot="electric_field_lines.png")  # 画像として保存
