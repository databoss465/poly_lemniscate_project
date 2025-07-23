#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

// nvcc -arch=sm_89 -Xcompiler -fPIC -shared monte_carlo.cu -o libcuda_mc.so

// Evaluates the polynomial on the GPU
__device__ double norm_eval (const cuDoubleComplex* coeffs, int degree, cuDoubleComplex z) {
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i <= degree; ++i) {
        result = cuCadd(cuCmul(result, z), coeffs[i]);
    }
    double norm_sq = cuCreal(result) * cuCreal(result) + cuCimag(result) * cuCimag(result);
    return sqrt(norm_sq);
}

// CUDA Kernel for Monte Carlo estimation
__global__ void  monte_carlo_kernel (
    cuDoubleComplex* coeffs, int degree, double x_min,
    double x_max, double y_min, double y_max, int n_pts,
    unsigned long long* inside_count, unsigned int seed) {

        // Unique thread index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_pts) return;

        // Random Point
        curandState state;
        curand_init(seed, idx, 0, &state);
        double x = x_min + (x_max - x_min) * curand_uniform(&state);
        double y = y_min + (y_max - y_min) * curand_uniform(&state);
        cuDoubleComplex z = make_cuDoubleComplex(x, y);

        // Evaluate polynomial and check membership
        double val = norm_eval(coeffs, degree, z);

        if (val < 1.0) {
            atomicAdd(inside_count, 1);
        }
}

//Host Wrapper
extern "C" double monte_carlo_estimate_cuda (
    const double* roots_re, const double* roots_im, int degree,
    double x_min, double x_max, double y_min, double y_max, int n_pts, int n_threads) {

    // Build Polynomial (Host)
    std::vector<std::complex<double>> coeffs = {1.0};
    for (int i = 0; i < degree; ++i) {
        std::complex<double> root(roots_re[i], roots_im[i]);
        std::vector<std::complex<double>> new_coeffs(coeffs.size() + 1, 0.0);
        for (size_t j = 0; j < coeffs.size(); ++j) {
            new_coeffs[j] += -root * coeffs[j];
            new_coeffs[j + 1] += coeffs[j];
        }
        coeffs = std::move(new_coeffs);
    }

    // Allocate device memory (Device)
    cuDoubleComplex* d_coeffs;
    cudaMalloc(&d_coeffs, coeffs.size() * sizeof(cuDoubleComplex));
    std::vector<cuDoubleComplex> coeffs_cuda(coeffs.size());
    for (int i = 0; i < coeffs.size(); ++i) 
        coeffs_cuda[i] = make_cuDoubleComplex(coeffs[i].real(), coeffs[i].imag());
    // Copies the memory from coeffs.size() startting from d_coeffs(host) to coeffs_cuda(device)
    cudaMemcpy(d_coeffs, coeffs_cuda.data(), coeffs.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    unsigned long long* d_in;
    cudaMalloc(&d_in, sizeof(unsigned long long));
    cudaMemset(d_in, 0, sizeof(unsigned long long));

    // Launch Kernel
    int threads_per_block = n_threads;
    int gridsize = (n_pts + threads_per_block - 1) / threads_per_block;
    monte_carlo_kernel<<<gridsize, threads_per_block>>>(
        d_coeffs, degree, x_min, x_max, y_min, y_max, n_pts, 
        d_in, 465);
    cudaDeviceSynchronize();

    // Copy result back to host
    unsigned long long inside_points;
    cudaMemcpy(&inside_points, d_in, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_coeffs);
    cudaFree(d_in);

    double total_area = (x_max - x_min) * (y_max - y_min);
    return total_area * (static_cast<double>(inside_points) / n_pts);
}
    

