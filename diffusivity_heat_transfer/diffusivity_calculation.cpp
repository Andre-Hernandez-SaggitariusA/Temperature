#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <omp.h>

// Compilar:
// g++ -O3 -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -fopenmp diffusivity_calculation.cpp -o difusividad$(python3-config --extension-suffix)

namespace py = pybind11;

int xyz(int Nx, int Ny, int Nz, int i, int j, int k){

	int ijk = k * (Ny + 1) * (Nx + 1) + j * (Nx + 1) + i;
	return ijk;
}

void actualizar(int Nx, int Ny, int Nz, double px, double py, double pz, py::array_t<double> u_py, py::array_t<double> u_new_py, int hilos){
		
	py::buffer_info u_new_b = u_new_py.request();
	py::buffer_info u_b = u_py.request();
	
	double *u_new = static_cast<double *>(u_new_b.ptr);
	double *u = static_cast<double *>(u_b.ptr);
	
	pybind11::gil_scoped_release release;
	
	#pragma omp parallel num_threads(hilos)
	{
	
	#pragma omp for collapse(3)
	for(int k = 1; k < Nz; k++){
		for(int j = 1; j < Ny; j++){
			for(int i = 1; i < Nx; i++){
			
			int central = xyz(Nx, Ny, Nz, i, j, k);
			int v_ip = xyz(Nx, Ny, Nz, i+1, j, k);
			int v_im = xyz(Nx, Ny, Nz, i-1, j, k);
			int v_jp = xyz(Nx, Ny, Nz, i, j+1, k);
			int v_jm = xyz(Nx, Ny, Nz, i, j-1, k);
			int v_kp = xyz(Nx, Ny, Nz, i, j, k+1);
			int v_km = xyz(Nx, Ny, Nz, i, j, k-1);
			
			// Interior
			u_new[central] = (1 - 2*px - 2*py - 2*pz)*u[central]
				+ px*(u[v_ip] + u[v_im])
				+ py*(u[v_jp] + u[v_jm])
				+ pz*(u[v_kp] + u[v_km]);
			
			}
		}	
	}
	
	#pragma omp for collapse(2)
	for(int j = 1; j < Ny; j++){
		for(int i = 1; i < Nx; i++){
			
			int central_0 = xyz(Nx, Ny, Nz, i, j, 0);
			int v_ip_0 = xyz(Nx, Ny, Nz, i+1, j, 0);
			int v_im_0 = xyz(Nx, Ny, Nz, i-1, j, 0);
			int v_jp_0 = xyz(Nx, Ny, Nz, i, j+1, 0);
			int v_jm_0 = xyz(Nx, Ny, Nz, i, j-1, 0);
			int v_k_0 = xyz(Nx, Ny, Nz, i, j, 1);
			
			// Base
				
			u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
				+ px*(u[v_ip_0] + u[v_im_0])
				+ py*(u[v_jp_0] + u[v_jm_0])
				+ 2*pz*u[v_k_0];
				
			int central_Nz = xyz(Nx, Ny, Nz, i, j, Nz);
			int v_ip_Nz = xyz(Nx, Ny, Nz, i+1, j, Nz);
			int v_im_Nz = xyz(Nx, Ny, Nz, i-1, j, Nz);
			int v_jp_Nz = xyz(Nx, Ny, Nz, i, j+1, Nz);
			int v_jm_Nz = xyz(Nx, Ny, Nz, i, j-1, Nz);
			int v_k_Nz = xyz(Nx, Ny, Nz, i, j, Nz-1);
				
			// Tapa	
			
			u_new[central_Nz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nz]
				+ px*(u[v_ip_Nz] + u[v_im_Nz])
				+ py*(u[v_jp_Nz] + u[v_jm_Nz])
				+ 2*pz*u[v_k_Nz];
		}
	}
	
	#pragma omp for collapse(2)
	for(int k = 1; k < Nz; k++){
		for(int i = 1; i < Nx; i++){
		
			int central_0 = xyz(Nx, Ny, Nz, i, 0, k);
			int v_ip_0 = xyz(Nx, Ny, Nz, i+1, 0, k);
			int v_im_0 = xyz(Nx, Ny, Nz, i-1, 0, k);
			int v_kp_0 = xyz(Nx, Ny, Nz, i, 0, k+1);
			int v_km_0 = xyz(Nx, Ny, Nz, i, 0, k-1);
			int v_j_0 = xyz(Nx, Ny, Nz, i, 1, k);
			
			// Cara Trasera
			
			u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
				+ px*(u[v_ip_0] + u[v_im_0])
				+ 2*py*u[v_j_0]
				+ pz*(u[v_kp_0] + u[v_km_0]);
			
			int central_Ny = xyz(Nx, Ny, Nz, i, Ny, k);
			int v_ip_Ny = xyz(Nx, Ny, Nz, i+1, Ny, k);
			int v_im_Ny = xyz(Nx, Ny, Nz, i-1, Ny, k);
			int v_kp_Ny = xyz(Nx, Ny, Nz, i, Ny, k+1);
			int v_km_Ny = xyz(Nx, Ny, Nz, i, Ny, k-1);
			int v_j_Ny = xyz(Nx, Ny, Nz, i, Ny-1, k);
			
			// Cara Posterior
			
			u_new[central_Ny] = (1 - 2*px - 2*py - 2*pz)*u[central_Ny]
				+ px*(u[v_ip_Ny] + u[v_im_Ny])
				+ 2*py*u[v_j_Ny]
				+ pz*(u[v_kp_Ny] + u[v_km_Ny]);
				
		}
	}
	
	#pragma omp for collapse(2)
	for(int k = 1; k < Nz; k++){
		for(int j = 1; j < Ny; j++){
		
			int central_0 = xyz(Nx, Ny, Nz, 0, j, k);
			int v_jp_0 = xyz(Nx, Ny, Nz, 0, j+1, k);
			int v_jm_0 = xyz(Nx, Ny, Nz, 0, j-1, k);
			int v_kp_0 = xyz(Nx, Ny, Nz, 0, j, k+1);
			int v_km_0 = xyz(Nx, Ny, Nz, 0, j, k-1);
			int v_i_0 = xyz(Nx, Ny, Nz, 1, j, k);
		
			// Cara Izquierda	
			
			u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
				+ 2*px*u[v_i_0]
				+ py*(u[v_jp_0] + u[v_jm_0])
				+ pz*(u[v_kp_0] + u[v_km_0]);
			
			int central_Nx = xyz(Nx, Ny, Nz, Nx, j, k);
			int v_jp_Nx = xyz(Nx, Ny, Nz, Nx, j+1, k);
			int v_jm_Nx = xyz(Nx, Ny, Nz, Nx, j-1, k);
			int v_kp_Nx = xyz(Nx, Ny, Nz, Nx, j, k+1);
			int v_km_Nx = xyz(Nx, Ny, Nz, Nx, j, k-1);
			int v_i_Nx = xyz(Nx, Ny, Nz, Nx-1, j, k);
			
			// Cara Derecha
			
			u_new[central_Nx] = (1 - 2*px - 2*py - 2*pz)*u[central_Nx]
				+ 2*px*u[v_i_Nx]
				+ py*(u[v_jp_Nx] + u[v_jm_Nx])
				+ pz*(u[v_kp_Nx] + u[v_km_Nx]);
				
		}
	}
	
	#pragma omp for
	for(int i = 1; i < Nx; i++){
		
		int central_0 = xyz(Nx, Ny, Nz, i, 0, 0);
		int v_ip_0 = xyz(Nx, Ny, Nz, i+1, 0, 0);
		int v_im_0 = xyz(Nx, Ny, Nz, i-1, 0, 0);
		int v_j_0 = xyz(Nx, Ny, Nz, i, 1, 0);
		int v_k_0 = xyz(Nx, Ny, Nz, i, 0, 1);
		
		// Arista trasera inferior
		
		u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
			+ px*(u[v_ip_0] + u[v_im_0])
			+ 2*py*u[v_j_0]
			+ 2*pz*u[v_k_0];
		
		int central_Ny = xyz(Nx, Ny, Nz, i, Ny, 0);
		int v_ip_Ny = xyz(Nx, Ny, Nz, i+1, Ny, 0);
		int v_im_Ny = xyz(Nx, Ny, Nz, i-1, Ny, 0);
		int v_j_Ny = xyz(Nx, Ny, Nz, i, Ny-1, 0);
		int v_k_Ny = xyz(Nx, Ny, Nz, i, Ny, 1);
		
		// Arista posterior inferior
		
		u_new[central_Ny] = (1 - 2*px - 2*py - 2*pz)*u[central_Ny]
			+ px*(u[v_ip_Ny] + u[v_im_Ny])
			+ 2*py*u[v_j_Ny]
			+ 2*pz*u[v_k_Ny];
			
		int central_Nz = xyz(Nx, Ny, Nz, i, 0, Nz);
		int v_ip_Nz = xyz(Nx, Ny, Nz, i+1, 0, Nz);
		int v_im_Nz = xyz(Nx, Ny, Nz, i-1, 0, Nz);
		int v_j_Nz = xyz(Nx, Ny, Nz, i, 1, Nz);
		int v_k_Nz = xyz(Nx, Ny, Nz, i, 0, Nz-1);
			
		// Arista trasera superior
			
		u_new[central_Nz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nz]
			+ px*(u[v_ip_Nz] + u[v_im_Nz])
			+ 2*py*u[v_j_Nz]
			+ 2*pz*u[v_k_Nz];
		
		int central_Nyz = xyz(Nx, Ny, Nz, i, Ny, Nz);
		int v_ip_Nyz = xyz(Nx, Ny, Nz, i+1, Ny, Nz);
		int v_im_Nyz = xyz(Nx, Ny, Nz, i-1, Ny, Nz);
		int v_j_Nyz = xyz(Nx, Ny, Nz, i, Ny-1, Nz);
		int v_k_Nyz = xyz(Nx, Ny, Nz, i, Ny, Nz-1);
		
		// Arista posterior superior
		
		u_new[central_Nyz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nyz]
			+ px*(u[v_ip_Nyz] + u[v_im_Nyz])
			+ 2*py*u[v_j_Nyz]
			+ 2*pz*u[v_k_Nyz];
	}
	
	#pragma omp for
	for(int j = 1; j < Ny; j++){
		
		int central_0 = xyz(Nx, Ny, Nz, 0, j, 0);
		int v_jp_0 = xyz(Nx, Ny, Nz, 0, j+1, 0);
		int v_jm_0 = xyz(Nx, Ny, Nz, 0, j-1, 0);
		int v_i_0 = xyz(Nx, Ny, Nz, 1, j, 0);
		int v_k_0 = xyz(Nx, Ny, Nz, 0, j, 1);
		
		// Arista inferior izquierda
		
		u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
			+ 2*px*u[v_i_0]
			+ py*(u[v_jp_0] + u[v_jm_0])
			+ 2*pz*u[v_k_0];
		
		int central_Nx = xyz(Nx, Ny, Nz, Nx, j, 0);
		int v_jp_Nx = xyz(Nx, Ny, Nz, Nx, j+1, 0);
		int v_jm_Nx = xyz(Nx, Ny, Nz, Nx, j-1, 0);
		int v_i_Nx = xyz(Nx, Ny, Nz, Nx-1, j, 0);
		int v_k_Nx = xyz(Nx, Ny, Nz, Nx, j, 1);
		
		// Arista inferior derecha
		
		u_new[central_Nx] = (1 - 2*px - 2*py - 2*pz)*u[central_Nx]
			+ 2*px*u[v_i_Nx]
			+ py*(u[v_jp_Nx] + u[v_jm_Nx])
			+ 2*pz*u[v_k_Nx];
			
		int central_Nz = xyz(Nx, Ny, Nz, 0, j, Nz);
		int v_jp_Nz = xyz(Nx, Ny, Nz, 0, j+1, Nz);
		int v_jm_Nz = xyz(Nx, Ny, Nz, 0, j-1, Nz);
		int v_i_Nz = xyz(Nx, Ny, Nz, 1, j, Nz);
		int v_k_Nz = xyz(Nx, Ny, Nz, 0, j, Nz-1);	
		
		// Arista superior izquierda
		
		u_new[central_Nz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nz]
			+ 2*px*u[v_i_Nz]
			+ py*(u[v_jp_Nz] + u[v_jm_Nz])
			+ 2*pz*u[v_k_Nz];
		
		int central_Nxz = xyz(Nx, Ny, Nz, Nx, j, Nz);
		int v_jp_Nxz = xyz(Nx, Ny, Nz, Nx, j+1, Nz);
		int v_jm_Nxz = xyz(Nx, Ny, Nz, Nx, j-1, Nz);
		int v_i_Nxz = xyz(Nx, Ny, Nz, Nx-1, j, Nz);
		int v_k_Nxz = xyz(Nx, Ny, Nz, Nx, j, Nz-1);
		
		// Arista superior derecha
		
		u_new[central_Nxz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nxz]
			+ 2*px*u[v_i_Nxz]
			+ py*(u[v_jp_Nxz] + u[v_jm_Nxz])
			+ 2*pz*u[v_k_Nxz];
	}
	
	#pragma omp for
	for(int k = 1; k < Nz; k++){
		
		int central_0 = xyz(Nx, Ny, Nz, 0, 0, k);
		int v_kp_0 = xyz(Nx, Ny, Nz, 0, 0, k+1);
		int v_km_0 = xyz(Nx, Ny, Nz, 0, 0, k-1);
		int v_i_0 = xyz(Nx, Ny, Nz, 1, 0, k);
		int v_j_0 = xyz(Nx, Ny, Nz, 0, 1, k);
		
		// Arista izquierda trasera
		
		u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
			+ 2*px*u[v_i_0]
			+ 2*py*u[v_j_0]
			+ pz*(u[v_kp_0] + u[v_km_0]);
		
		int central_Nx = xyz(Nx, Ny, Nz, Nx, 0, k);
		int v_kp_Nx = xyz(Nx, Ny, Nz, Nx, 0, k+1);
		int v_km_Nx = xyz(Nx, Ny, Nz, Nx, 0, k-1);
		int v_i_Nx = xyz(Nx, Ny, Nz, Nx-1, 0, k);
		int v_j_Nx = xyz(Nx, Ny, Nz, Nx, 1, k);
		
		// Arista derecha trasera
		
		u_new[central_Nx] = (1 - 2*px - 2*py - 2*pz)*u[central_Nx]
			+ 2*px*u[v_i_Nx]
			+ 2*py*u[v_j_Nx]
			+ pz*(u[v_kp_Nx] + u[v_km_Nx]);
			
		int central_Ny = xyz(Nx, Ny, Nz, 0, Ny, k);
		int v_kp_Ny = xyz(Nx, Ny, Nz, 0, Ny, k+1);
		int v_km_Ny = xyz(Nx, Ny, Nz, 0, Ny, k-1);
		int v_i_Ny = xyz(Nx, Ny, Nz, 1, Ny, k);
		int v_j_Ny = xyz(Nx, Ny, Nz, 0, Ny-1, k);
			
		// Arista izquierda posterior
		
		u_new[central_Ny] = (1 - 2*px - 2*py - 2*pz)*u[central_Ny]
			+ 2*px*u[v_i_Ny]
			+ 2*py*u[v_j_Ny]
			+ pz*(u[v_kp_Ny] + u[v_km_Ny]);
		
		int central_Nxy = xyz(Nx, Ny, Nz, Nx, Ny, k);
		int v_kp_Nxy = xyz(Nx, Ny, Nz, Nx, Ny, k+1);
		int v_km_Nxy = xyz(Nx, Ny, Nz, Nx, Ny, k-1);
		int v_i_Nxy = xyz(Nx, Ny, Nz, Nx-1, Ny, k);
		int v_j_Nxy = xyz(Nx, Ny, Nz, Nx, Ny-1, k);
		
		// Arista derecha posterior
		
		u_new[central_Nxy] = (1 - 2*px - 2*py - 2*pz)*u[central_Nxy]
			+ 2*px*u[v_i_Nxy]
			+ 2*py*u[v_j_Nxy]
			+ pz*(u[v_kp_Nxy] + u[v_km_Nxy]);
	}
	
	#pragma omp single
	{
	
	int central_0 = xyz(Nx, Ny, Nz, 0, 0, 0);
	int v_i_0 = xyz(Nx, Ny, Nz, 1, 0, 0);
	int v_j_0 = xyz(Nx, Ny, Nz, 0, 1, 0);
	int v_k_0 = xyz(Nx, Ny, Nz, 0, 0, 1);
	
	// Esquina izquierda inferior trasera
	
	u_new[central_0] = (1 - 2*px - 2*py - 2*pz)*u[central_0]
			+ 2*px*u[v_i_0]
			+ 2*py*u[v_j_0]
			+ 2*pz*u[v_k_0];
	
	int central_Nx = xyz(Nx, Ny, Nz, Nx, 0, 0);
	int v_i_Nx = xyz(Nx, Ny, Nz, Nx-1, 0, 0);
	int v_j_Nx = xyz(Nx, Ny, Nz, Nx, 1, 0);
	int v_k_Nx = xyz(Nx, Ny, Nz, Nx, 0, 1);
	
	// Esquina derecha inferior trasera
	
	u_new[central_Nx] = (1 - 2*px - 2*py - 2*pz)*u[central_Nx]
			+ 2*px*u[v_i_Nx]
			+ 2*py*u[v_j_Nx]
			+ 2*pz*u[v_k_Nx];
		
	int central_Ny = xyz(Nx, Ny, Nz, 0, Ny, 0);
	int v_i_Ny = xyz(Nx, Ny, Nz, 1, Ny, 0);
	int v_j_Ny = xyz(Nx, Ny, Nz, 0, Ny-1, 0);
	int v_k_Ny = xyz(Nx, Ny, Nz, 0, Ny, 1);
			
	// Esquina izquierda inferior posterior
	
	u_new[central_Ny] = (1 - 2*px - 2*py - 2*pz)*u[central_Ny]
			+ 2*px*u[v_i_Ny]
			+ 2*py*u[v_j_Ny]
			+ 2*pz*u[v_k_Ny];
	
	int central_Nz = xyz(Nx, Ny, Nz, 0, 0, Nz);
	int v_i_Nz = xyz(Nx, Ny, Nz, 1, 0, Nz);
	int v_j_Nz = xyz(Nx, Ny, Nz, 0, 1, Nz);
	int v_k_Nz = xyz(Nx, Ny, Nz, 0, 0, Nz-1);
	
	// Esquina izquierda superior trasera
	
	u_new[central_Nz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nz]
			+ 2*px*u[v_i_Nz]
			+ 2*py*u[v_j_Nz]
			+ 2*pz*u[v_k_Nz];
	
	int central_Nxy = xyz(Nx, Ny, Nz, Nx, Ny, 0);
	int v_i_Nxy = xyz(Nx, Ny, Nz, Nx-1, Ny, 0);
	int v_j_Nxy = xyz(Nx, Ny, Nz, Nx, Ny-1, 0);
	int v_k_Nxy = xyz(Nx, Ny, Nz, Nx, Ny, 1);
	
	// Esquina derecha inferior posterior
	
	u_new[central_Nxy] = (1 - 2*px - 2*py - 2*pz)*u[central_Nxy]
			+ 2*px*u[v_i_Nxy]
			+ 2*py*u[v_j_Nxy]
			+ 2*pz*u[v_k_Nxy];
	
	int central_Nxz = xyz(Nx, Ny, Nz, Nx, 0, Nz);
	int v_i_Nxz = xyz(Nx, Ny, Nz, Nx-1, 0, Nz);
	int v_j_Nxz = xyz(Nx, Ny, Nz, Nx, 1, Nz);
	int v_k_Nxz = xyz(Nx, Ny, Nz, Nx, 0, Nz-1);
	
	// Esquina derecha superior trasera
	
	u_new[central_Nxz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nxz]
			+ 2*px*u[v_i_Nxz]
			+ 2*py*u[v_j_Nxz]
			+ 2*pz*u[v_k_Nxz];
	
	int central_Nyz = xyz(Nx, Ny, Nz, 0, Ny, Nz);
	int v_i_Nyz = xyz(Nx, Ny, Nz, 1, Ny, Nz);
	int v_j_Nyz = xyz(Nx, Ny, Nz, 0, Ny-1, Nz);
	int v_k_Nyz = xyz(Nx, Ny, Nz, 0, Ny, Nz-1);
	
	// Esquina izquierda superior posterior
	
	u_new[central_Nyz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nyz]
			+ 2*px*u[v_i_Nyz]
			+ 2*py*u[v_j_Nyz]
			+ 2*pz*u[v_k_Nyz];
	
	int central_Nxyz = xyz(Nx, Ny, Nz, Nx, Ny, Nz);
	int v_i_Nxyz = xyz(Nx, Ny, Nz, Nx-1, Ny, Nz);
	int v_j_Nxyz = xyz(Nx, Ny, Nz, Nx, Ny-1, Nz);
	int v_k_Nxyz = xyz(Nx, Ny, Nz, Nx, Ny, Nz-1);
	
	// Esquina derecha superior posterior
	
	u_new[central_Nxyz] = (1 - 2*px - 2*py - 2*pz)*u[central_Nxyz]
			+ 2*px*u[v_i_Nxyz]
			+ 2*py*u[v_j_Nxyz]
			+ 2*pz*u[v_k_Nxyz];
	
	}	
	}
	
	pybind11::gil_scoped_acquire acquire;
	
}

PYBIND11_MODULE(difusividad, m) {
    m.doc() = "Módulo de difusividad en C++";
    m.def("actualizar", &actualizar, "Función que calcula la actualización difusiva");
}
