#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#pragma GCC optimize("O3","unroll-loops","inline")

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    bool is_big_matrix = ( n <= 500 ) ? 0 : 1;
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    //matrix size is A : n * m ; B : m * l ; result matrix C is n * l;
    //B should be col major for better cache use
    //size_t is the datatype for size/byte counts
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const size_t size_A = (size_t)n * (size_t)m;
    const size_t size_B = (size_t)m * (size_t)l;
    int *A = NULL, *B = NULL;
    if (posix_memalign((void**)&A, 64, size_A * sizeof(int)) != 0)
        A = NULL;
    if (posix_memalign((void**)&B, 64, size_B * sizeof(int)) != 0)
        B = NULL;
    //error handles 
    if (!A || !B) {
        free(A);
        free(B);
        fprintf(stderr, "malloc failed in construct_matrices on this rank\n");
        MPI_Abort(MPI_COMM_WORLD, 1);   // ← abort instead of return
    }
    if (world_rank == 0) {
        //copy a as row major
        memcpy(A, a_mat, size_A * sizeof(int));
        //b_mat is already read as column major in main.cc!
        memcpy(B, b_mat, size_B * sizeof(int));
    }
    *a_mat_ptr = A;
    *b_mat_ptr = B;

    MPI_Bcast(*a_mat_ptr, n * m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void small_matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    //initialize
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //use conditional assignment so whole function can see A and B
    /*
    int *A = (world_rank == 0) ? (int*)a_mat
                           : (int*)malloc((size_t)n * m * sizeof(int));
    int *B = (world_rank == 0) ? (int*)b_mat
                           : (int*)malloc((size_t)m * l * sizeof(int));
                           */
    int *A = (int*)a_mat;
    int *B = (int*)b_mat;
    //checks allocation error
    if ((world_rank != 0) && (!A || !B)) {
        fprintf(stderr, "malloc failed on rank %d\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    //all ranks have A and B, will compute their own part
    //MPI_Bcast(A, n * m, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(B, m * l, MPI_INT, 0, MPI_COMM_WORLD);
    int row_num = n / world_size;
    int extra_row = n % world_size;

    //some ranks need to do extra work because we cannot fully divide n, need left extra_row many ranks compute 1 more row.
    //if it is within extra range, starting point is rank*(row_num + 1), because previous ranks all have taken 1 extra row.
    //if it is out of extra range, starting point need to add up all extra.
    int row_start = (world_rank < extra_row) ? world_rank * (row_num + 1) : world_rank * row_num + extra_row;
    int my_row = (world_rank < extra_row) ? (row_num + 1) : row_num;
    int row_end = row_start + my_row;

    //prevents if n < rank
    //allocates my_row * l int, the part the rank need to calculate.
    int *C_local = (my_row > 0) ? (int*)malloc((size_t)my_row * l * sizeof(int)) : NULL;
    //fail allocation test
    if (my_row > 0 && !C_local) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for(int i = row_start; i < row_end; i++)
    {
        //process all row
        const int *A_row = A + (size_t)i * m;
        int *C_row = C_local + (size_t)(i - row_start) * l;
        //C is m by l;
        for (int j = 0; j < l; j++)
        {
            //B is col major
            const int *B_col = B + (size_t)j * m;
            long long acc = 0;
            //each element in one row/col
            for (int k = 0; k < m; ++k) {
                acc += (long long)A_row[k] * B_col[k];
            }
            C_row[j] = (int)acc;
        }
    }

    //handles broadcast
    std::vector<int> recv_counts, displs;
    if (world_rank == 0) {
        recv_counts.resize(world_size);
        displs.resize(world_size);
        int off = 0;
        for (int r = 0; r < world_size; ++r) {
            int r_rows = (r < extra_row) ? (row_num+1) : row_num;
            recv_counts[r] = r_rows * l; //how many elements from rank r
            displs[r]     = off;        //where in out_mat does rank r's element start
            off          += recv_counts[r];
        }
    }

    MPI_Gatherv(C_local, my_row * l, MPI_INT,
            (world_rank == 0 ? out_mat            : nullptr),
            (world_rank == 0 ? recv_counts.data() : nullptr),
            (world_rank == 0 ? displs.data()      : nullptr),
            MPI_INT, 0, MPI_COMM_WORLD);
    //clean up
    if (C_local) free(C_local);
}
void big_matrix_multiply(
    const int n, const int m, const int l, const int * __restrict__ a_mat, const int * __restrict__ b_mat, int * __restrict__ out_mat)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    int *A = (int*)a_mat;
    int *B = (int*)b_mat;


    int row_num = n / world_size;
    int extra_row = n % world_size;
    int row_start = (world_rank < extra_row) ? world_rank * (row_num + 1) : world_rank * row_num + extra_row;
    int my_row = (world_rank < extra_row) ? (row_num + 1) : row_num;

    int *C_local = (my_row > 0) ? (int*)malloc((size_t)my_row * l * sizeof(int)) : nullptr;
    if (my_row > 0 && !C_local) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    //we need to read from local c with tiling, so we need to set 0 here
    if (C_local) memset(C_local, 0, (size_t)my_row * l * sizeof(int));

    //parameter for tiling, can tune
    const int BK = 512;   //split m (shared by A B) to many k pieces each of size BK 
    const int BJ = 128;   //split j

    //A is size n * m, row-major, row i is: A[i * m + 0 .. i * m + (m-1)]
    //B is size m * l, column-major, column j is: B[j * m + 0 .. j * m + (m-1)]
    //loop acroess column in B
    for (int jj = 0; jj < l; jj += BJ) {
        const int J = (jj + BJ < l) ? BJ : (l - jj);

        //For each row i (A)
        for (int i = 0; i < my_row; ++i) {
            const int global_i = row_start + i;

            int *C_row_jj = C_local + (size_t)i * l + jj; // length J

            //For each column j in this tile
            for (int j = 0; j < J; ++j) {
                //sum accumulates dot product across many K.
                int sum = C_row_jj[j];

                //Sweep K dimension in tiles, accumulating
                for (int total_k = 0; total_k < m; total_k += BK) {
                    const int k_len = (total_k + BK < m) ? BK : (m - total_k);

                    const int *A_row_kk =
                        A + (size_t)global_i * m + total_k;  // row i, K-panel
                    const int *B_col_panel =
                        B + (size_t)(jj + j) * m + total_k;  // col (jj+j), K-panel

                    for (int k = 0; k < k_len; ++k) {
                        sum += A_row_kk[k] * B_col_panel[k];
                    }
                }

                C_row_jj[j] = sum;
            }
        }
    }


    std::vector<int> recv_counts, displs;
    if (world_rank == 0) {
        recv_counts.resize(world_size);
        displs.resize(world_size);
        int off = 0;
        for (int r = 0; r < world_size; ++r) {
            const int r_rows = (r < extra_row) ? (row_num+1) : row_num;
            recv_counts[r] = r_rows * l;
            displs[r]      = off;
            off           += recv_counts[r];
        }
    }

    MPI_Gatherv(C_local, my_row * l, MPI_INT,
                (world_rank == 0 ? out_mat            : nullptr),
                (world_rank == 0 ? recv_counts.data() : nullptr),
                (world_rank == 0 ? displs.data()      : nullptr),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (C_local) free(C_local);
}


void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    bool is_big_matrix = ( n <= 500 ) ? 0 : 1;
    big_matrix_multiply(n, m, l, a_mat, b_mat, out_mat);
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    free(a_mat);
    free(b_mat);
}

