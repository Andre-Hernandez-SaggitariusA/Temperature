#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <omp.h>

// Compilar:
// g++ -O3 -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -fopenmp diffusivity_calculation.cpp -o difusividad$(python3-config --extension-suffix)

namespace py = pybind11;

void actualizar(int Nx,
		int Ny,
		int Nz,
		double px,
		double py,
		double pz,
		py::array_t<double> u_py,
		py::array_t<double> u_new_py,
		int hilos){
		
	py::buffer_info u_new_b = u_new_py.request();
	py::buffer_info u_b = u_py.request();
	
	int s_k = (Ny + 1) * (Nx + 1);
	int s_j = (Nx + 1);
	double c = 1 - 2*px - 2*py - 2*pz;
	
	double *u_new = static_cast<double *>(u_new_b.ptr);
	double *u = static_cast<double *>(u_b.ptr);
	
	pybind11::gil_scoped_release release;
	
	#pragma omp parallel num_threads(hilos)
	{
	
	#pragma omp for collapse(2) nowait
	for(int k = 1; k < Nz; k++){
		for(int j = 1; j < Ny; j++){
		
			int base = k*s_k + j*s_j;
		
			#pragma omp simd
			for(int i = 1; i < Nx; i++){
			
				int p = base + i;
			
				// Interior
				
				u_new[p] = c*u[p]
					+ px*(u[p + 1] + u[p - 1])
					+ py*(u[p + s_j] + u[p - s_j])
					+ pz*(u[p + s_k] + u[p - s_k]);
			}
		}	
	}
	
	#pragma omp for collapse(2) nowait
	for(int j = 1; j < Ny; j++){
		for(int i = 1; i < Nx; i++){
			
			// Base
				
			u_new[0*s_k + j*s_j + i] = c*u[0*s_k + j*s_j + i]
				+ px*(u[0*s_k + j*s_j + (i+1)] + u[0*s_k + j*s_j + (i-1)])
				+ py*(u[0*s_k + (j+1)*s_j + i] + u[0*s_k + (j-1)*s_j + i])
				+ 2*pz*u[1*s_k + j*s_j + i];
				
			// Tapa	
			
			u_new[Nz*s_k + j*s_j + i] = c*u[Nz*s_k + j*s_j + i]
				+ px*(u[Nz*s_k + j*s_j + (i+1)] + u[Nz*s_k + j*s_j + (i-1)])
				+ py*(u[Nz*s_k + (j+1)*s_j + i] + u[Nz*s_k + (j-1)*s_j + i])
				+ 2*pz*u[(Nz-1)*s_k + j*s_j + i];
		}
	}
	
	#pragma omp for collapse(2) nowait
	for(int k = 1; k < Nz; k++){
		for(int i = 1; i < Nx; i++){
			
			// Cara Trasera
			
			u_new[k*s_k + 0*s_j + i] = c*u[k*s_k + 0*s_j + i]
				+ px*(u[k*s_k + 0*s_j + (i+1)] + u[k*s_k + 0*s_j + (i-1)])
				+ 2*py*u[k*s_k + 1*s_j + i]
				+ pz*(u[(k+1)*s_k + 0*s_j + i] + u[(k-1)*s_k + 0*s_j + i]);
			
			// Cara Posterior
			
			u_new[k*s_k + Ny*s_j + i] = c*u[k*s_k + Ny*s_j + i]
				+ px*(u[k*s_k + Ny*s_j + (i+1)] + u[k*s_k + Ny*s_j + (i-1)])
				+ 2*py*u[k*s_k + (Ny-1)*s_j + i]
				+ pz*(u[(k+1)*s_k + Ny*s_j + i] + u[(k-1)*s_k + Ny*s_j + i]);
				
		}
	}
	
	#pragma omp for collapse(2) nowait
	for(int k = 1; k < Nz; k++){
		for(int j = 1; j < Ny; j++){
		
			// Cara Izquierda	
			
			u_new[k*s_k + j*s_j + 0] = c*u[k*s_k + j*s_j + 0]
				+ 2*px*u[k*s_k + j*s_j + 1]
				+ py*(u[k*s_k + (j+1)*s_j + 0] + u[k*s_k + (j-1)*s_j + 0])
				+ pz*(u[(k+1)*s_k + j*s_j + 0] + u[(k-1)*s_k + j*s_j + 0]);
			
			// Cara Derecha
			
			u_new[k*s_k + j*s_j + Nx] = c*u[k*s_k + j*s_j + Nx]
				+ 2*px*u[k*s_k + j*s_j + (Nx-1)]
				+ py*(u[k*s_k + (j+1)*s_j + Nx] + u[k*s_k + (j-1)*s_j + Nx])
				+ pz*(u[(k+1)*s_k + j*s_j + Nx] + u[(k-1)*s_k + j*s_j + Nx]);
				
		}
	}
	
	#pragma omp for nowait
	for(int i = 1; i < Nx; i++){
		
		// Arista trasera inferior
		
		u_new[0*s_k + 0*s_j + i] = c*u[0*s_k + 0*s_j + i]
			+ px*(u[0*s_k + 0*s_j + (i+1)] + u[0*s_k + 0*s_j + (i-1)])
			+ 2*py*u[0*s_k + 1*s_j + i]
			+ 2*pz*u[1*s_k + 0*s_j + i];
		
		// Arista posterior inferior
		
		u_new[0*s_k + Ny*s_j + i] = c*u[0*s_k + Ny*s_j + i]
			+ px*(u[0*s_k + Ny*s_j + (i+1)] + u[0*s_k + Ny*s_j + (i-1)])
			+ 2*py*u[0*s_k + (Ny-1)*s_j + i]
			+ 2*pz*u[1*s_k + Ny*s_j + i];
			
		// Arista trasera superior
			
		u_new[Nz*s_k + 0*s_j + i] = c*u[Nz*s_k + 0*s_j + i]
			+ px*(u[Nz*s_k + 0*s_j + (i+1)] + u[Nz*s_k + 0*s_j + (i-1)])
			+ 2*py*u[Nz*s_k + 1*s_j + i]
			+ 2*pz*u[(Nz-1)*s_k + 0*s_j + i];
		
		// Arista posterior superior
		
		u_new[Nz*s_k + Ny*s_j + i] = c*u[Nz*s_k + Ny*s_j + i]
			+ px*(u[Nz*s_k + Ny*s_j + (i+1)] + u[Nz*s_k + Ny*s_j + (i-1)])
			+ 2*py*u[Nz*s_k + (Ny-1)*s_j + i]
			+ 2*pz*u[(Nz-1)*s_k + Ny*s_j + i];
	}
	
	#pragma omp for nowait
	for(int j = 1; j < Ny; j++){
		
		// Arista inferior izquierda
		
		u_new[0*s_k + j*s_j + 0] = c*u[0*s_k + j*s_j + 0]
			+ 2*px*u[0*s_k + j*s_j + 1]
			+ py*(u[0*s_k + (j+1)*s_j + 0] + u[0*s_k + (j-1)*s_j + 0])
			+ 2*pz*u[1*s_k + j*s_j + 0];
		
		// Arista inferior derecha
		
		u_new[0*s_k + j*s_j + Nx] = c*u[0*s_k + j*s_j + Nx]
			+ 2*px*u[0*s_k + j*s_j + (Nx-1)]
			+ py*(u[0*s_k + (j+1)*s_j + Nx] + u[0*s_k + (j-1)*s_j + Nx])
			+ 2*pz*u[1*s_k + j*s_j + Nx];	
		
		// Arista superior izquierda
		
		u_new[Nz*s_k + j*s_j + 0] = c*u[Nz*s_k + j*s_j + 0]
			+ 2*px*u[Nz*s_k + j*s_j + 1]
			+ py*(u[Nz*s_k + (j+1)*s_j + 0] + u[Nz*s_k + (j-1)*s_j + 0])
			+ 2*pz*u[(Nz-1)*s_k + j*s_j + 0];
		
		// Arista superior derecha
		
		u_new[Nz*s_k + j*s_j + Nx] = c*u[Nz*s_k + j*s_j + Nx]
			+ 2*px*u[Nz*s_k + j*s_j + (Nx-1)]
			+ py*(u[Nz*s_k + (j+1)*s_j + Nx] + u[Nz*s_k + (j-1)*s_j + Nx])
			+ 2*pz*u[(Nz-1)*s_k + j*s_j + Nx];
	}
	
	#pragma omp for nowait
	for(int k = 1; k < Nz; k++){
		
		// Arista izquierda trasera
		
		u_new[k*s_k + 0*s_j + 0] = c*u[k*s_k + 0*s_j + 0]
			+ 2*px*u[k*s_k + 0*s_j + 1]
			+ 2*py*u[k*s_k + 1*s_j + 0]
			+ pz*(u[(k+1)*s_k + 0*s_j + 0] + u[(k-1)*s_k + 0*s_j + 0]);
		
		// Arista derecha trasera
		
		u_new[k*s_k + 0*s_j + Nx] = c*u[k*s_k + 0*s_j + Nx]
			+ 2*px*u[k*s_k + 0*s_j + (Nx-1)]
			+ 2*py*u[k*s_k + 1*s_j + Nx]
			+ pz*(u[(k+1)*s_k + 0*s_j + Nx] + u[(k-1)*s_k + 0*s_j + Nx]);
			
		// Arista izquierda posterior
		
		u_new[k*s_k + Ny*s_j + 0] = c*u[k*s_k + Ny*s_j + 0]
			+ 2*px*u[k*s_k + Ny*s_j + 1]
			+ 2*py*u[k*s_k + (Ny-1)*s_j + 0]
			+ pz*(u[(k+1)*s_k + Ny*s_j + 0] + u[(k-1)*s_k + Ny*s_j + 0]);
		
		// Arista derecha posterior
		
		u_new[k*s_k + Ny*s_j + Nx] = c*u[k*s_k + Ny*s_j + Nx]
			+ 2*px*u[k*s_k + Ny*s_j + (Nx-1)]
			+ 2*py*u[k*s_k + (Ny-1)*s_j + Nx]
			+ pz*(u[(k+1)*s_k + Ny*s_j + Nx] + u[(k-1)*s_k + Ny*s_j + Nx]);
	}
	
	#pragma omp single nowait
	{
	
	// Esquina izquierda inferior trasera
	
	u_new[0*s_k + 0*s_j + 0] = c*u[0*s_k + 0*s_j + 0]
			+ 2*px*u[0*s_k + 0*s_j + 1]
			+ 2*py*u[0*s_k + 1*s_j + 0]
			+ 2*pz*u[1*s_k + 0*s_j + 0];

	// Esquina derecha inferior trasera
	
	u_new[0*s_k + 0*s_j + Nx] = c*u[0*s_k + 0*s_j + Nx]
			+ 2*px*u[0*s_k + 0*s_j + (Nx-1)]
			+ 2*py*u[0*s_k + 1*s_j + Nx]
			+ 2*pz*u[1*s_k + 0*s_j + Nx];
			
	// Esquina izquierda inferior posterior
	
	u_new[0*s_k + Ny*s_j + 0] = c*u[0*s_k + Ny*s_j + 0]
			+ 2*px*u[0*s_k + Ny*s_j + 1]
			+ 2*py*u[0*s_k + (Ny-1)*s_j + 0]
			+ 2*pz*u[1*s_k + Ny*s_j + 0];
	
	// Esquina izquierda superior trasera
	
	u_new[Nz*s_k + 0*s_j + 0] = c*u[Nz*s_k + 0*s_j + 0]
			+ 2*px*u[Nz*s_k + 0*s_j + 1]
			+ 2*py*u[Nz*s_k + 1*s_j + 0]
			+ 2*pz*u[(Nz-1)*s_k + 0*s_j + 0];
	
	// Esquina derecha inferior posterior
	
	u_new[0*s_k + Ny*s_j + Nx] = c*u[0*s_k + Ny*s_j + Nx]
			+ 2*px*u[0*s_k + Ny*s_j + (Nx-1)]
			+ 2*py*u[0*s_k + (Ny-1)*s_j + Nx]
			+ 2*pz*u[1*s_k + Ny*s_j + Nx];
	
	// Esquina derecha superior trasera
	
	u_new[Nz*s_k + 0*s_j + Nx] = c*u[Nz*s_k + 0*s_j + Nx]
			+ 2*px*u[Nz*s_k + 0*s_j + (Nx-1)]
			+ 2*py*u[Nz*s_k + 1*s_j + Nx]
			+ 2*pz*u[(Nz-1)*s_k + 0*s_j + Nx];
	
	// Esquina izquierda superior posterior
	
	u_new[Nz*s_k + Ny*s_j + 0] = c*u[Nz*s_k + Ny*s_j + 0]
			+ 2*px*u[Nz*s_k + Ny*s_j + 1]
			+ 2*py*u[Nz*s_k + (Ny-1)*s_j + 0]
			+ 2*pz*u[(Nz-1)*s_k + Ny*s_j + 0];
	
	// Esquina derecha superior posterior
	
	u_new[Nz*s_k + Ny*s_j + Nx] = c*u[Nz*s_k + Ny*s_j + Nx]
			+ 2*px*u[Nz*s_k + Ny*s_j + (Nx-1)]
			+ 2*py*u[Nz*s_k + (Ny-1)*s_j + Nx]
			+ 2*pz*u[(Nz-1)*s_k + Ny*s_j + Nx];
	
	}	
	}
}

PYBIND11_MODULE(difusividad, m) {
    m.doc() = "Módulo de difusividad en C++";
    m.def("actualizar", &actualizar, "Función que calcula la actualización difusiva");
}
