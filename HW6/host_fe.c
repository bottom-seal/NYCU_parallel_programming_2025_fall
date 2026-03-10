
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "host_fe.h"
#include "helper.h"

#define GROUP_SIZE_H 8
#define GROUP_SIZE_W 16

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;
    int filter_size = filter_width * filter_width;
    int img_size = image_height * image_width;

    int local_height = 2 * GROUP_SIZE_H + 2 * (filter_width / 2);
    int local_width  = 2 * GROUP_SIZE_W + 2 * (filter_width / 2);
    int local_size   = local_height * local_width;

    //command queue
    cl_command_queue queue =
        clCreateCommandQueue(*context, *device, 0, NULL);

    //device buffers
    cl_mem filter_buf = clCreateBuffer(
        *context,
        CL_MEM_USE_HOST_PTR,
        filter_size * sizeof(float),
        filter,
        NULL
    );

    cl_mem input_buf  = clCreateBuffer(
        *context,
        CL_MEM_USE_HOST_PTR,
        img_size * sizeof(float),
        input_image,
        NULL
    );

    cl_mem output_buf = clCreateBuffer(
        *context,
        CL_MEM_WRITE_ONLY,
        img_size * sizeof(float),
        NULL,
        NULL
    );

    //initialize kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    //parameters for kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&filter_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&input_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buf);
    clSetKernelArg(kernel, 3, sizeof(int)   , (void*)&filter_width);
    clSetKernelArg(kernel, 4, sizeof(int)   , (void*)&image_height);
    clSetKernelArg(kernel, 5, sizeof(int)   , (void*)&image_width);
    clSetKernelArg(kernel, 6, sizeof(int)   , (void*)&local_width);
    //clSetKernelArg(kernel, 7, sizeof(int)   , (void*)&local_height);   
    clSetKernelArg(kernel, 7, sizeof(float) * local_size, NULL);

    size_t local[2] = {
        GROUP_SIZE_W,
        GROUP_SIZE_H
    };

    //ceil(a / b) = (a + b - 1) / b
    //change from (image_width, image_height) to 
    //(image_width/local_width/2, image_height/local_height/2)
    //about 1/4
    int groups_x = (image_width  + (2 * GROUP_SIZE_W - 1)) / (2 * GROUP_SIZE_W);
    int groups_y = (image_height + (2 * GROUP_SIZE_H - 1)) / (2 * GROUP_SIZE_H);

    size_t global[2] = {
        (size_t)groups_x * GROUP_SIZE_W,
        (size_t)groups_y * GROUP_SIZE_H
    };

    //launch kernel
    clEnqueueNDRangeKernel(
        queue, kernel, 2, NULL, global, local, 0, NULL, NULL
    );

    clEnqueueReadBuffer(
        queue,
        output_buf,
        CL_TRUE,
        0,
        img_size * sizeof(float),
        output_image,
        0,
        NULL,
        NULL
    );

    //clReleaseMemObject(devFilterBuffer);
    //clReleaseMemObject(devInputBuffer);
    //clReleaseMemObject(devOutputBuffer);
    //clReleaseKernel(kernel);
    //clReleaseCommandQueue(queue);
}
