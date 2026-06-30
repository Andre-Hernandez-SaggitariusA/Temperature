# Propagación de Calor 3D

# Librerías importadas

import numpy as np
import pyvista as pv
import propagacion
import time

# Parámetros físicos y numéricos

Lx = 1.0
Ly = 1.0
Lz = 1.0

alpha = 1
tau = 0.5
v = np.sqrt(alpha / tau)

# Partición de malla

Nx = 60
Ny = 60
Nz = 60

dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

# Parámetros de avance temporal

dt = 0.95 * ((dx*dy*dz)/(v*np.sqrt(dy*dy*dz*dz + dx*dx*dz*dz + dx*dx*dy*dy)))
t_final = 5
snapshot_interval = 1

# Pesos para cálculos

pa = ((4*dt*dt)/(2*tau + dt))*(tau/(dt*dt) - alpha/(dx*dx) - alpha/(dy*dy) - alpha/(dz*dz))
po = (2*tau - dt)/(2*tau + dt)
px = (2*dt*dt*alpha)/((2*tau + dt)*dx*dx)
py = (2*dt*dt*alpha)/((2*tau + dt)*dy*dy)
pz = (2*dt*dt*alpha)/((2*tau + dt)*dz*dz)

# Generación de la matriz vacía

u = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()
u_new = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()
u_old = np.zeros((Nx + 1, Ny + 1, Nz + 1)).flatten()

# Saltos de eje para moverse entre la matriz 1D

s_k = (Ny + 1) * (Nx + 1)
s_j = (Nx + 1)

# Condiciones de generación de calor

def aplicar_calor(u,t):
	
	u[15*s_k + 15*s_j + 15] = 1
	u[45*s_k + 45*s_j + 45] = 1
	
# Evolución con el tiempo

snapshots = []
tiempos = []
hilos = 8

aplicar_calor(u,0)
u_old[:] = u[:]

t0 = time.perf_counter()

for n in range(int(t_final/dt) + 1):

	# Actualizar la matriz con funcion de calor.cpp
	
	propagacion.actualizar(Nx, Ny, Nz, pa, po, px, py, pz, u, u_old, u_new, hilos)

	u, u_old, u_new = u_new, u, u_old
	
	# Aplicar calor
	
	if (n > 0) and (n % 20 == 0):
		aplicar_calor(u,n*dt)
		
	# Guardar snapshots

	if n % snapshot_interval == 0 :
		snapshots.append(u.astype(np.float32, copy=True))
		tiempos.append(n*dt)
	
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
	clim=[0, maximo],
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
