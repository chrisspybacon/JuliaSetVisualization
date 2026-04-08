#include "msvc_defines.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

#define HIP_CHECK(status)						\
if (status != hipSuccess)						\
{									\
	cerr << hipGetErrorString(status) << endl;	                \
	exit(0);						        \
}

void plot(size_t canvas_size_x, size_t canvas_size_y, double start_x, double start_y, string path);

__global__ void calc_iterations(double start_x, double start_y, const size_t canvas_size_x, const size_t canvas_size_y, const double step, const size_t iterations, unsigned int* r_gpu, unsigned int* g_gpu, unsigned int* b_gpu);

int main()
{
    const size_t iterations = 10000;
    const double step = 0.00069444444444;
    const double size_x = 1.77777777777777;
    const double size_y = 1;
    const size_t canvas_size_x = (double)size_x * (1 / step); // px of the generated picture
    const size_t canvas_size_y = (double)size_y * (1 / step);
    double start_x = 0 - size_x / 2, start_y = 0 + size_y / 2; // upper left corner of the picture
    string path = "fractal.ppm";
    plot(canvas_size_x, canvas_size_y, start_x, start_y, path);
    return 0;
}

void plot(size_t canvas_size_x, size_t canvas_size_y, double start_x, double start_y, string path)
{
    ofstream output(path);
    output << "P3\n";
    output << canvas_size_x << " " << canvas_size_y << "\n";
    output << "255\n";
    unsigned int* r_ptr = (unsigned int*)malloc(canvas_size_x * canvas_size_y * sizeof(unsigned int));
    unsigned int* g_ptr = (unsigned int*)malloc(canvas_size_x * canvas_size_y * sizeof(unsigned int));
    unsigned int* b_ptr = (unsigned int*)malloc(canvas_size_x * canvas_size_y * sizeof(unsigned int));
    unsigned int* r_gpu = 0;
    unsigned int* g_gpu = 0;
    unsigned int* b_gpu = 0;
    HIP_CHECK(hipMalloc((void**)&r_gpu, canvas_size_x * canvas_size_y * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&g_gpu, canvas_size_x * canvas_size_y * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&b_gpu, canvas_size_x * canvas_size_y * sizeof(unsigned int)));
    const size_t THREADS_PER_BLOCK_X = 32;
    const size_t THREADS_PER_BLOCK_Y = 32;
    const size_t BLOCKS_PER_GRID_X = ceil((double)canvas_size_x / THREADS_PER_BLOCK_X);
    const size_t BLOCKS_PER_GRID_Y = ceil((double)canvas_size_y / THREADS_PER_BLOCK_Y);
    calc_iterations<<<dim3(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0>>>(start_x, start_y, canvas_size_x, canvas_size_y, step, iterations, r_gpu, g_gpu, b_gpu);
    HIP_CHECK(hipMemcpy(r_ptr, r_gpu, canvas_size_x * canvas_size_y * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(g_ptr, g_gpu, canvas_size_x * canvas_size_y * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b_ptr, b_gpu, canvas_size_x * canvas_size_y * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(r_gpu));
    HIP_CHECK(hipFree(g_gpu));
    HIP_CHECK(hipFree(b_gpu));
    cout << "saving picture..." << endl;
    for (size_t i = 0;i < canvas_size_y;++i)
    {
        for (size_t j = 0;j < canvas_size_x;++j)
        {
            output << r_ptr[i * canvas_size_x + j] << " " << g_ptr[i * canvas_size_x + j] << " " << b_ptr[i * canvas_size_x + j] << "  ";
        }
        output << "\n";
    }
    free(r_ptr);
    free(g_ptr);
    free(b_ptr);
}

__global__ void calc_iterations(double start_x, double start_y, const size_t canvas_size_x, const size_t canvas_size_y, const double step, const size_t iterations, unsigned int* r_gpu, unsigned int* g_gpu, unsigned int* b_gpu)
{
    if ((blockDim.x * blockIdx.x + threadIdx.x < (unsigned int)canvas_size_x) && (blockDim.y * blockIdx.y + threadIdx.y < (unsigned int)canvas_size_y))
    {
        int x_pos = blockDim.x * blockIdx.x + threadIdx.x;
        int y_pos = blockDim.y * blockIdx.y + threadIdx.y;
        double x0 = -0.391;
        double y0 = -0.587;
        double x = start_x + (double)x_pos * step, y = start_y - (double)y_pos * step; // components of zn, i.e. of the nth iteration
        double temp_x;
        size_t temp = 0;
        while ((x * x + y * y <= 4) && (temp < iterations))
        {
            temp_x = x * x - y * y + x0; // calculating the real component of the nth iteration
            y = 2 * y * x + y0; // calculating the imaginary component of the nth iteration
            x = temp_x;
            ++temp;
        }
        if (temp == iterations)
            r_gpu[y_pos * canvas_size_x + x_pos] = g_gpu[y_pos * canvas_size_x + x_pos] = b_gpu[y_pos * canvas_size_x + x_pos] = 0;
        else {
            double t = log((double)temp + 1) / log((double)iterations);
            if (t < 0.16) 
            {
                // Dark blue to light blue
                r_gpu[y_pos * canvas_size_x + x_pos] = 0;
                g_gpu[y_pos * canvas_size_x + x_pos] = (int)(t / 0.16 * 255);
                b_gpu[y_pos * canvas_size_x + x_pos] = (int)(64 + (t / 0.16 * 191));
            }
            else if (t < 0.42) {
                // Light blue to cyan
                r_gpu[y_pos * canvas_size_x + x_pos] = 0;
                g_gpu[y_pos * canvas_size_x + x_pos] = (int)(255 * (0.5 + 0.5 * (t - 0.16) / 0.26));
                b_gpu[y_pos * canvas_size_x + x_pos] = 255;
            }
            else if (t < 0.6425) {
                // Cyan to yellow
                r_gpu[y_pos * canvas_size_x + x_pos] = (int)(255 * (t - 0.42) / 0.2225);
                g_gpu[y_pos * canvas_size_x + x_pos] = 255;
                b_gpu[y_pos * canvas_size_x + x_pos] = (int)(255 * (1.0 - (t - 0.42) / 0.2225));
            }
            else if (t < 0.8575) {
                // Yellow to red
                r_gpu[y_pos * canvas_size_x + x_pos] = 255;
                g_gpu[y_pos * canvas_size_x + x_pos] = (int)(255 * (1.0 - (t - 0.6425) / 0.215));
                b_gpu[y_pos * canvas_size_x + x_pos] = 0;
            }
            else {
                // Red to dark red
                r_gpu[y_pos * canvas_size_x + x_pos] = (int)(255 * (1.0 - 0.5 * (t - 0.8575) / 0.1425));
                g_gpu[y_pos * canvas_size_x + x_pos] = 0;
                b_gpu[y_pos * canvas_size_x + x_pos] = 0;
            }
        }
    }
}
