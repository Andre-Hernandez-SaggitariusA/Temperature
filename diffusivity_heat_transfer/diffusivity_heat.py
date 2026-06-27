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

Nx = 50
Ny = 50
Nz = 50

dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

# Parámetros de avance temporal

dt = 0.95 * (dx*dx*dy*dy*dz*dz)/(2*alpha*(dy*dy*dz*dz + dx*dx*dz*dz + dx*dx*dy*dy))
t_final = 5000
snapshot_interval = 5

# Pesos de los cálculos

px = (alpha*dt)/(dx*dx)
py = (alpha*dt)/(dy*dy)
pz = (alpha*dt)/(dz*dz)

# Hilos de paralelización OMP

hilos = 6

# Generación de la matriz vacía

u = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()
u_new = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()

# Función para moverse por índices de matriz aplanada

def xyz(i, j, k, Nx=Nx, Ny=Ny, Nz=Nz):

	ijk = k * (Ny + 1) * (Nx + 1) + j * (Nx + 1) + i
	return ijk

# Condiciones de generación de calor

def aplicar_calor(u,t):
	
	for i in range(0, Nx+1):
		u[xyz(i, 5, 5)] += 5000

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

print(t1-t0)

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

# Actualización de temperaturas en animación

for i in range(len(snapshots)):
	
	point_cloud["calor"] = snapshots[i]
	plotter.write_frame()

plotter.show()
plotter.close()
