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
    long long count = 0;
    long long local_tosses = tosses/world_size;
    for(long long toss = 0; toss < local_tosses; toss++) {
        double x = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
        double y = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
        double distance_squared = x * x + y * y;
        if (x * x + y * y <= 1)
            count++;
    }

    MPI_Status status;
    MPI_Request request;
    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Send(&count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        //long long *bufs     = (long long*)malloc((size_t)(world_size-1) * sizeof(long long));
        //MPI_Request *reqs   = (MPI_Request*)malloc((size_t)(world_size-1) * sizeof(MPI_Request));
        MPI_Request requests[world_size - 1];
        long long buffer[world_size - 1];
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        for (int source = 1; source < world_size; source++) {
             MPI_Irecv(&buffer[source-1], 1, MPI_LONG_LONG, source, 0, MPI_COMM_WORLD, &requests[source-1]);
        }
        pi_result = count;
        // MPI_Request requests[];
        // MPI_Waitall();
        MPI_Waitall(world_size-1, requests, MPI_STATUSES_IGNORE);
        for (int i = 0; i < world_size-1; ++i) {
            pi_result += buffer[i];                     // bufs[i] holds rank (i+1)’s hits
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

