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
    srand(seed);

    MPI_Status status;
    MPI_Request request;

    // TODO: use MPI_Reduce
    if (world_rank > 0)
    {
        long long count = 0;
        // TODO: handle workers
        long long local_tosses = tosses/world_size;    
        for(long long toss = 0; toss < local_tosses; toss++) {
            double x = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
            double y = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
            double distance_squared = x * x + y * y;
            if (distance_squared <= 1.0) {
                count++;
            }
        }
        MPI_Reduce(&count, NULL,1, MPI_LONG_LONG, MPI_SUM,0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        long long *counts = malloc((size_t)world_size * sizeof(long long));
        long long count = 0;
        long long result = 0;
        for(long long toss = 0; toss < (tosses/world_size); toss++) {
            double x = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
            double y = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
            double distance_squared = x * x + y * y;
            if (distance_squared <= 1.0) {
                count++;
            }
        }
        // TODO: main
        MPI_Reduce(&count, &result, 1, MPI_LONG_LONG, MPI_SUM,0, MPI_COMM_WORLD);
        pi_result = result;
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

