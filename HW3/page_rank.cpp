#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <cstdio>
#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs
    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:
    */
    // initialization: see example code above
    double *score_old = (double*) malloc(sizeof(double) * nnodes);
    double *score_new = (double*) malloc(sizeof(double) * nnodes);
    bool *is_dangling = (bool*) malloc(sizeof(bool) * nnodes);
    if (!score_old || !score_new || !is_dangling) {
        std::fprintf(stderr, "Allocation failed (nnodes=%d)\n", nnodes);
        std::free(is_dangling);
        std::free(score_new);
        std::free(score_old);
        return; // function is void
    }
    bool converged = false;
    double global_diff = 0;
    double dangling_sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
        score_old[i] = equal_prob;
        //find all nodes with no outgoing edge.
        is_dangling[i] = (outgoing_size(g,i) == 0);
    }

    while (!converged) {
        #pragma omp parallel
        {
            // compute score_new[vi] for all nodes vi:
            //score_new[vi] = sum over all nodes vj reachable from incoming edges
            //                { score_old[vj] / number of edges leaving vj  }
            #pragma omp for
            for (int i = 0; i < nnodes; i++)
            {
                score_new[i] = 0.0;
                int start_edge = g->incoming_starts[i]; 
                int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
                //neighbor is an index into imcoming edges array, the range is the nodes that points to node i.
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                {
                    //one of node that points to i
                    int incoming = g->incoming_edges[neighbor];
                    score_new[i] += score_old[incoming]/outgoing_size(g, incoming);
                }
            }

            //score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;
            #pragma omp for
            for (int i = 0; i < nnodes; i++)
            {
                score_new[i] = (damping * score_new[i]) + (1.0-damping) / nnodes;
            }

            //score_new[vi] += sum over all nodes v in graph with no outgoing edges
            //                    { damping * score_old[v] / nnodes }
            //we dont need to find all nodes with no outgoing edge in each iterations, they stay the same for a graph
            #pragma omp single
            {
                dangling_sum = 0;
            }
            #pragma omp for reduction(+:dangling_sum)
            for (int i = 0; i < nnodes; i++)
            {
                if (is_dangling[i])
                  dangling_sum +=  score_old[i];
            }
            #pragma omp single
            {
                dangling_sum =  damping * dangling_sum / nnodes; 
            }
            #pragma omp for
            for (int i = 0; i < nnodes; i++)
            {
                score_new[i] += dangling_sum;
            }
            // compute how much per-node scores have changed
            // quit once algorithm has converged

            //global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
            #pragma omp single
            {
                global_diff = 0;
            }
            #pragma omp for reduction(+:global_diff)
            for (int i = 0; i < nnodes; i++)
            {
                global_diff +=  fabs(score_new[i] - score_old[i]);
            }
            //#pragma omp single
            //{
            //    converged = (global_diff < convergence);
            //    double* tmp = score_old;
            //    score_old = score_new;
            //    score_new = tmp;
            //}
            #pragma omp single
            {
                converged = (global_diff < convergence);
            }

            #pragma omp for
            for (int i = 0; i < nnodes; ++i)
            {
                score_old[i] = score_new[i];
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = score_old[i];
    }
    free(score_old);
    free(score_new);
    free(is_dangling);
}

