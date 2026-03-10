#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ void mandel_kernel(int* output, float lower_x, float lower_y, float step_x, float step_y, int res_x, int res_y, int max_iterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= res_x || j >= res_y)
        return;
    
    float x = lower_x + ((float)i * step_x);
    float y = lower_y + ((float)j * step_y);

    float z_re = x, z_im = y;

    int  iter    = 0;
    bool escaped = false;
    while (iter + 16 <= max_iterations) {
        float zr2 = z_re * z_re;
        float zi2 = z_im * z_im;
        float new_re, new_im;

        // ---- step 1 ----
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 2 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 3 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 4 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 5 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 6 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 7 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 8 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 9 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 10 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 11 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 12 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 13 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 14 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 15 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;

        // ---- step 16 ----
        zr2 = z_re * z_re;
        zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.f) { escaped = true; break; }
        new_re = zr2 - zi2;
        new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
        iter++;
    }

    //if not dividable by 16
    if (!escaped) {
        while (iter < max_iterations) {
            float zr2 = z_re * z_re;
            float zi2 = z_im * z_im;
            if (zr2 + zi2 > 4.f)
                break;
            float new_re = zr2 - zi2;
            float new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            iter++;
        }
    }
    output[j*res_x + i] = iter;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,//x0
             float upper_y,//y0
             float lower_x,//x1
             float lower_y,//y1
             int *img,//output image
             int res_x,//width
             int res_y,//height
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;//dx
    float step_y = (upper_y - lower_y) / (float)res_y;//dy

    //load stuff to device memory
    int *output;
    int size = res_x * res_y * sizeof(int);
    cudaHostRegister(img, size, cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&output, img, 0);
    //set up execution configuration
    //slide setup
    /*
    dim3 dimBlock(res_x, res_y);//if res_x * res_y <= maxThreadsPerBlock
    dim3 dimGrid(1, 1);
    */
    //recommanded setup
    dim3 dimBlock(8, 8);  // 256 threads per block
    dim3 dimGrid(
        (res_x + dimBlock.x - 1) / dimBlock.x,
        (res_y + dimBlock.y - 1) / dimBlock.y
    );
    //GeForce GTX 1060 6GB can use up to 32*32

    //invoke kernel function
    mandel_kernel<<<dimGrid, dimBlock>>>(output, lower_x, lower_y, step_x, step_y, res_x, res_y, max_iterations);

    //read output from device memory
    cudaDeviceSynchronize();
    cudaHostUnregister(img);
}


