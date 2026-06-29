# Difusividad Calor 3D

# Librerías importadas

import numpy as np
import pyvista as pv
import difusividad
import time

# Parámetros físicos

Lx = 1.0
Ly = 1.0
Lz = 1.0

alpha = 2.2e-5

# Partición de malla

Nx = 100
Ny = 100
Nz = 100

dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

# Parámetros de avance temporal

dt = 0.95 * (dx*dx*dy*dy*dz*dz)/(2*alpha*(dy*dy*dz*dz + dx*dx*dz*dz + dx*dx*dy*dy))
t_final = 8000
snapshot_interval = 100

# Pesos de los cálculos

px = (alpha*dt)/(dx*dx)
py = (alpha*dt)/(dy*dy)
pz = (alpha*dt)/(dz*dz)

# Hilos de paralelización OMP

hilos = 8

# Generación de la matriz vacía

u = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()
u_new = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()

# Saltos de eje para moverse entre la matriz 1D

s_k = (Ny + 1) * (Nx + 1)
s_j = (Nx + 1)

# Condiciones de generación de calor

def aplicar_calor(u, t):
	
	for i in range(0, Nx+1):
		u[25*s_k + 25*s_j + i] += 10

# Evolución con el tiempo

snapshots = []
tiempos = []

t0 = time.perf_counter()

for n in range(int(t_final/dt) + 1):

	# Aplicar calor
	
	aplicar_calor(u,n*dt)

	# Guardar snapshots

	if n % snapshot_interval == 0 :
		snapshots.append(np.copy(u))
		tiempos.append(n*dt)

	# Actualizar la matriz con funcion de calor.cpp
	
	difusividad.actualizar(Nx, Ny, Nz, px, py, pz, u, u_new, hilos)

	u, u_new = u_new, u

t1 = time.perf_counter()

print(f"Tiempo de cálculo: {(t1-t0):.3f} s")

# Generación de malla

x, y, z = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), np.arange(Nz+1), indexing="ij")

x = x.flatten(order="F")
y = y.flatten(order="F")
z = z.flatten(order="F")

puntos = np.vstack((x, y, z)).T

# Parámetros de barra informativa

barra = {
	"title": "Temperatura",
	"vertical": True,
	"position_x": 0.90,
	"position_y": 0.15,
	"width": 0.04,
	"height": 0.70
	}

# Temperatura máxima y mínima alcanzada

minimo = np.min(snapshots)
maximo = np.max(snapshots)

# Generación de video

t0 = time.perf_counter()

point_cloud = pv.PolyData(puntos.astype(np.float32))

point_cloud["calor"] = snapshots[0]
plotter = pv.Plotter()

plotter.open_movie("difusividad_termica.mp4", framerate=5)

# Características de la simulación

plotter.add_mesh(
	point_cloud,
	style="points",
	point_size=15.0,
	render_points_as_spheres=False,
	scalars="calor",
	cmap="hot",
	clim=[minimo, maximo],
	opacity=[0.0, 1.0],
	scalar_bar_args=barra
	)
	
plotter.view_isometric()
plotter.add_axes()

t1 = time.perf_counter()

print(f"Tiempo de generación de simulación: {(t1-t0):.3f} s")


# Actualización de temperaturas en animación

t0 = time.perf_counter()

for i in range(len(snapshots)):
	
	point_cloud["calor"] = snapshots[i]
	plotter.write_frame()

t1 = time.perf_counter()

print(f"Tiempo de simulación: {(t1-t0):.3f} s")

plotter.show()
plotter.close()
