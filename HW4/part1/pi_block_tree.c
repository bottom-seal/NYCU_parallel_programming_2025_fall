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
    // TODO: binary tree redunction
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
    //each process has it own count
    int divider = 2;
    int rem = 1;
    for(int step = 1; step < world_size; step <<= 1)
    {
        if (world_rank % divider == 0)
        {
            long long temp;
            MPI_Recv(&temp, 1, MPI_LONG_LONG, world_rank+rem, 0, MPI_COMM_WORLD, &status);
            count = count + temp;
        }
        else if (world_rank % divider == rem)
        {
            MPI_Send(&count, 1, MPI_LONG_LONG, world_rank-rem, 0, MPI_COMM_WORLD);
            break;
        }
        divider *= 2;
        rem *=2;
    }
        
    
    
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * count / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

