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

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned base = (unsigned) time(NULL);
    unsigned seed = base + (unsigned)(world_rank * 123);
    //srand(seed);

    MPI_Status status;
    MPI_Request request;

    // TODO: use MPI_Gather
    if (world_rank > 0)
    {
        long long count = 0;
        long long local_tosses = tosses/world_size;
        // TODO: handle workers
        for(long long toss = 0; toss < local_tosses; toss++) {
            double x = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
            double y = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
            double distance_squared = x * x + y * y;
            if (distance_squared <= 1.0) {
                count++;
            }
        }
        MPI_Gather(&count,  1, MPI_LONG_LONG, NULL,  0, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        long long *counts = malloc((size_t)world_size * sizeof(long long));
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
        // TODO: main
        MPI_Gather(&count,  1, MPI_LONG_LONG, counts,  1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        pi_result = 0;
        for (int i = 0; i < world_size; ++i) {
            pi_result += counts[i];                     // bufs[i] holds rank (i+1)’s hits
        }
    }
    if (world_rank == 0)
    {
        // TODO: PI result
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

