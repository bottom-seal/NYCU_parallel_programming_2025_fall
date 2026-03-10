#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cmath>
const int GROUP_ROWS = 4;  // e.g., 2, 4, 8, ...

__global__ void mandel_kernel(int* output, size_t pitch, float lower_x, float lower_y, float step_x, float step_y, int res_x, int res_y, int max_iterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j_base = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= res_x || j_base >= res_y)
        return;
    
    float x = lower_x + ((float)i * step_x);

    int cnt;
    for(int slice_cnt = 0; slice_cnt < GROUP_ROWS; ++slice_cnt)
    {  
        int j = j_base + slice_cnt * blockDim.y * gridDim.y;
        if (j >= res_y)
            break;
        
        float y = lower_y + (float)j * step_y;
        
        float z_re = x, z_im = y;
        for (cnt = 0; cnt < max_iterations; ++cnt)
        {

            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = (z_re * z_re) - (z_im * z_im);
            float new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
        }
        char *row_base = (char *)output + j * pitch;
        int  *row      = (int *)row_base;
        row[i] = cnt;
    }
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
    int size = res_x * res_y * sizeof(int);
    
    //host allocates pinned memory
    int *host_output;
    cudaHostAlloc(&host_output, size, cudaHostAllocDefault);

    //device allocates pinned memory
    int *output;
    size_t pitch = 0;
    size_t row_bytes = res_x * sizeof(int);
    cudaMallocPitch(&output, &pitch, row_bytes, res_y);
    //pitch device memeory

    //set up execution configuration
    //slide setup
    /*
    dim3 dimBlock(res_x, res_y);//if res_x * res_y <= maxThreadsPerBlock
    dim3 dimGrid(1, 1);
    */
    //recommanded setup
    dim3 dimBlock(16, 16);  // 256 threads per block
    dim3 dimGrid(
        (res_x + dimBlock.x - 1) / dimBlock.x,
        ceil((float)(res_y + dimBlock.y - 1) / (float)dimBlock.y / (float)GROUP_ROWS)
    );
    //GeForce GTX 1060 6GB can use up to 32*32

    //invoke kernel function
    mandel_kernel<<<dimGrid, dimBlock>>>(output, pitch, lower_x, lower_y, step_x, step_y, res_x, res_y, max_iterations);

    //read output from device memory
    cudaMemcpy2D(
        host_output,        // dst
        row_bytes,          // dst pitch (tightly packed)
        output,              // src
        pitch,              // src pitch (from cudaMallocPitch)
        row_bytes,          // width of useful data per row
        res_y,              // rows
        cudaMemcpyDeviceToHost
    );

    memcpy(img, host_output, size);
    cudaFree(output);
    cudaFreeHost(host_output);
}

