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
    int cnt;
    for (cnt = 0; cnt < max_iterations; ++cnt)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    output[j*res_x + i] = cnt;
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
    cudaMalloc(&output, size);

    //set up execution configuration
    //slide setup
    /*
    dim3 dimBlock(res_x, res_y);//if res_x * res_y <= maxThreadsPerBlock
    dim3 dimGrid(1, 1);
    */
    //recommanded setup
    dim3 dimBlock(16, 16);  // 256 threads per block
    dim3 dimGrid(
        (res_x + dimBlock.x - 1) / dimBlock.x,   // ceil(res_x / 16)
        (res_y + dimBlock.y - 1) / dimBlock.y    // ceil(res_y / 16)
    );
    //GeForce GTX 1060 6GB can use up to 32*32

    //invoke kernel function
    mandel_kernel<<<dimGrid, dimBlock>>>(output, lower_x, lower_y, step_x, step_y, res_x, res_y, max_iterations);

    //read output from device memory
    cudaMemcpy(img, output, size, cudaMemcpyDeviceToHost);
    cudaFree(output);
}

