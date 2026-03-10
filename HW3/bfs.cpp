#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    //Loop over all vertices in the current frontier 
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < frontier->count; i++)
    {
        // Take one vertex from the frontier
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            //actual neighbor vertex current node points out to
            int outgoing = g->outgoing_edges[neighbor];
            //if this neighbor is not visited yet
            //CRITICAL PART
            int newdist = distances[node] + 1;
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, newdist)) {
                    //set distance 1 more than current node
                    //distances[outgoing] = distances[node] + 1;
                
                    //add this node to next round's frontier
                    //COUNT IS CRITICAL PART TOO
                    int index;
                    #pragma omp atomic capture //need capture else can only apply to 1 statement
                    index = new_frontier->count++;
                    new_frontier->vertices[index] = outgoing;
                }
                
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}
//use extra flags for speed up (reads 1 bit)
void bottom_up_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances, bool *check_flag)
{   
    //indicates vertex j is in frontier
    #pragma omp parallel for
    for(int j = 0; j < frontier->count; j++)
    {
        check_flag[frontier->vertices[j]] = 1;
    }
    //for(each vertex v in graph)
    #pragma omp parallel for schedule(dynamic, 1024)
    for(int i = 0; i < g -> num_nodes; i++)
    {
        // if(v has not been visited)
        if(distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
            //check i's neighbor
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                //incoming is one of the neighbor node
                int incoming = g->incoming_edges[neighbor];
                if (check_flag[incoming])
                {
                    distances[i] = distances[incoming] + 1;
                    int index;
                    #pragma omp atomic capture
                    index = new_frontier->count++;
                    new_frontier->vertices[index] = i;
                    break;
                }
            }
        }
    }
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool* frontier_flag = new bool[graph->num_nodes];

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);
        std::memset(frontier_flag, 0, graph->num_nodes * sizeof(bool));
        bottom_up_step(graph, frontier, new_frontier, sol->distances, frontier_flag);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    delete[] frontier_flag;
    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool* frontier_flag = new bool[graph->num_nodes];
    long long m_u = 0;
    for (int v = 0; v < graph->num_nodes; ++v)
            m_u += incoming_size(graph, v);
    m_u -= incoming_size(graph, ROOT_NODE_ID);
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif
        //m_f : the number of outgoing edges to examine if you run Top-Down this level.
        long long m_f = 0;
        //m_u : the number of incoming edges incident to still-unvisited vertices—a proxy for how much work Bottom-Up would do this level. 
        for(int i = 0; i < frontier->count; i++)
        {
            m_f += outgoing_size(graph, frontier->vertices[i]);
        }

        vertex_set_clear(new_frontier);
        if(m_f > m_u / 14)
        {
            std::memset(frontier_flag, 0, graph->num_nodes * sizeof(bool));
            bottom_up_step(graph, frontier, new_frontier, sol->distances, frontier_flag);
        }
        else
        {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }

        for (int i = 0; i < new_frontier->count; ++i)
            m_u = (m_u - incoming_size(graph, new_frontier->vertices[i]) > 0) ? m_u - incoming_size(graph, new_frontier->vertices[i]) : 0;
#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    if(frontier_flag)
        delete[] frontier_flag;
    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

