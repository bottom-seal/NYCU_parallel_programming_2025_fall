#define GROUP_SIZE_H 8
#define GROUP_SIZE_W 16

__attribute__((reqd_work_group_size(GROUP_SIZE_W, GROUP_SIZE_H, 1)))
__kernel void convolution(
    __constant float * restrict filter,
    __global  const float * restrict input_image,
    __global        float * restrict output_image,
    int filter_width,
    int image_height,
    int image_width,
    int local_width,
    //int local_height,
    __local  float *local_buf
)
{
    //need tile + filter bounds size to compute a tile
    int half_filter_size = filter_width / 2;

    //gets position
    int group_x    = get_group_id(0);
    int group_y    = get_group_id(1);
    int local_x   = get_local_id(0);
    int local_y   = get_local_id(1);
    int local_w_size = get_local_size(0);//GROUP_SIZE_W
    int local_h_size = get_local_size(1);//GROUP_SIZE_H

    //gets location wrt to the whole map
    //top left of the block
    int block_start_x = group_x * (2 * local_w_size);
    int block_start_y = group_y * (2 * local_h_size);

    //every thread computes a 2x2 block (adjacent)
    //x, y strided by 2.
    int j0 = block_start_x + 2 * local_x;
    int j1 = j0 + 1;
    int i0 = block_start_y + 2 * local_y;
    int i1 = i0 + 1;

    //tile size
    int tile_height   = 2 * local_h_size + 2 * half_filter_size;
    int tile_width    = local_width;
    int tile_size     = tile_width * tile_height;

    //
    int local_size    = local_w_size * local_h_size;
    int local_id      = local_y * local_w_size + local_x;

    //every thread loads a value to local memory
    //top-left of needed data in this tile
    int base_img_x = block_start_x - half_filter_size;
    int base_img_y = block_start_y - half_filter_size;

    //bool if whole tile is in image
    int tile_inside =
        (base_img_x >= 0) &&
        (base_img_y >= 0) &&
        (base_img_x + tile_width  - 1 < image_width) &&
        (base_img_y + tile_height - 1 < image_height);

    //strided loading so each thread has balanced load
    //thread 0 loads indexes: 0, local_size, 2*local_size
    //thread 1 loads indexes: 1, local_size + 1, 2*local_size+1
    //...
    if (tile_inside) {
        //skips bound check
        for (int t = local_id; t < tile_size; t += local_size) {
            int tile_x = t % tile_width;
            int tile_y = t / tile_width;
            int img_x = base_img_x + tile_x;
            int img_y = base_img_y + tile_y;
            local_buf[t] = input_image[img_y * image_width + img_x];
        }
    } else {
        for (int t = local_id; t < tile_size; t += local_size) {
            int tile_x = t % tile_width;
            int tile_y = t / tile_width;
            int img_x = base_img_x + tile_x;
            int img_y = base_img_y + tile_y;

            float val = 0.0f;
            if (img_x >= 0 && img_x < image_width && img_y >= 0 && img_y < image_height) {
                val = input_image[img_y * image_width + img_x];
            }
            local_buf[t] = val;
        }
    }
    //wait everyone load
    barrier(CLK_LOCAL_MEM_FENCE);

    //4-register convolution
    //can skip if all 4 pixels are not in image
    if (j0 >= image_width && j1 >= image_width)
        return;
    if (i0 >= image_height && i1 >= image_height)
        return;

    //get coordinate in local buf
    int local_j0 = 2 * local_x + half_filter_size;
    int local_j1 = local_j0 + 1;
    int local_i0 = 2 * local_y + half_filter_size;
    int local_i1 = local_i0 + 1;

    float sum00 = 0.0f;  // (i0, j0)
    float sum01 = 0.0f;  // (i0, j1)
    float sum10 = 0.0f;  // (i1, j0)
    float sum11 = 0.0f;  // (i1, j1)

    #pragma unroll
    //loop for y
    for (int k = -half_filter_size; k <= half_filter_size; ++k) {
        int ty0 = local_i0 + k;
        int ty1 = local_i1 + k;

        //row * tile_width + col
        //starting index of row ty0 and ty1
        int shared_row_base0 = ty0 * tile_width;
        int shared_row_base1 = ty1 * tile_width;

        int filter_row = (k + half_filter_size) * filter_width;

        #pragma unroll
        //loop for x
        for (int l = -half_filter_size; l <= half_filter_size; ++l) {
            int tx0 = local_j0 + l;  // col for j0
            int tx1 = tx0 + 1;      // col for j1

            float w = filter[filter_row + (l + half_filter_size)];

            float v00 = local_buf[shared_row_base0 + tx0];
            float v01 = local_buf[shared_row_base0 + tx1];
            float v10 = local_buf[shared_row_base1 + tx0];
            float v11 = local_buf[shared_row_base1 + tx1];

            sum00 = mad(w, v00, sum00);
            sum01 = mad(w, v01, sum01);
            sum10 = mad(w, v10, sum10);
            sum11 = mad(w, v11, sum11);
        }
    }

    //store results
    if (i0 < image_height && j0 < image_width)
        output_image[i0 * image_width + j0] = sum00;
    if (i0 < image_height && j1 < image_width)
        output_image[i0 * image_width + j1] = sum01;
    if (i1 < image_height && j0 < image_width)
        output_image[i1 * image_width + j0] = sum10;
    if (i1 < image_height && j1 < image_width)
        output_image[i1 * image_width + j1] = sum11;
}

