#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned base = (unsigned) time(NULL);
    unsigned seed = base + (unsigned)(world_rank * 123);
    //srand(seed);

    MPI_Status status;

    long long count = 0;
    long long local_tosses = tosses/world_size;    
    for(long long toss = 0; toss < local_tosses; toss++) {
        double x = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
        double y = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1.0) {
            count++;
        }
    }

    if (world_rank == 0)
    {
        long long *arr = NULL;
        // Main
        MPI_Alloc_mem((long)(2 * sizeof(long long)), MPI_INFO_NULL, (void *)&arr);
        arr[0] = 0;
        arr[1] = 0;
        MPI_Win_create( arr, (long)(2 * sizeof(long long)), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        long long test = 0;                               //displacement unit
        while(test != world_size-1)
        {
            MPI_Win_sync(win);
            test = arr[1];
        }
        pi_result = count + arr[0];
        MPI_Free_mem(arr);
    }
    else
    {
        // Workers
        long long one = 1;
        MPI_Win_create(NULL, 0, sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&count, 1, MPI_LONG_LONG,0   ,0     , 1, MPI_LONG_LONG, MPI_SUM, win);
        //                                     /dest/offset /how many entries
        MPI_Accumulate(&one, 1, MPI_LONG_LONG,0, 1, 1, MPI_LONG_LONG, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * pi_result / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

