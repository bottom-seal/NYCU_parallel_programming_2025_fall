#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "cg_impl.h"
#include "randdp.h"
#include <omp.h>

//---------------------------------------------------------------------
// Floaging point arrays here are named as in spec discussion of
// CG algorithm
//---------------------------------------------------------------------
//target is Az = x
//Compressed Sparse Row matrix:
//a[]: all nonzero values laid out row by row
//colidx[]: the column index for each nonzero in a[] (which column value is at)
//rowstr[]: rowstr(j) is the location of the first nonzero of row j
//col slice is for 1-D vectors, row slices is for matrix A, each thread gets a partition.
void conj_grad(const int colidx[],//column index
               const int rowstr[],//row pointer
               const double x[],//answer
               double z[],//solution
               const double a[],//matrix A
               double p[],//search direction
               double q[],//Ap, is size of N since NxN times N-vector is N vector
               double r[],//residual
               double *rnorm)
{
    int cgit, cgitmax = 25;
    //d = dot product accumulator
    //sum = accumulator for sum
    //rho = current squared residual
    //rho0 = previous rho, for beta
    //alpha = step lenght
    //beta = direction mix
    double d, sum, rho, rho0, alpha, beta;

    rho = 0.0;//current squared residual

    //initialize thread 


    //---------------------------------------------------------------------
    // Initialize the CG algorithm:
    //---------------------------------------------------------------------
    #pragma omp parallel for
    for (int j = 0; j < naa + 1; j++)
    {
        q[j] = 0.0;//Ap
        z[j] = 0.0;//sol
        r[j] = x[j];//residual = b (Az is 0)
        p[j] = r[j];//current search direction = residual
    }

    //reduction task
    //---------------------------------------------------------------------
    // rho = r.r
    // Now, obtain the norm of r: First, sum squares of r elements locally...
    //---------------------------------------------------------------------
    #pragma omp parallel for reduction (+:rho)
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        //rho = rho + r[j] * r[j];
        rho +=  r[j] * r[j];
    }

    //---------------------------------------------------------------------
    //---->
    // The conj grad iteration loop
    //---->
    //---------------------------------------------------------------------
    for (cgit = 1; cgit <= cgitmax; cgit++)
    //hardcap 25 iterations
    {
        #pragma omp parallel
        {
            //---------------------------------------------------------------------
            // q = A.p
            // The partition submatrix-vector multiply: use workspace w
            //---------------------------------------------------------------------
            //
            // NOTE: this version of the multiply is actually (slightly: maybe %5)
            //       faster on the sp2 on 16 nodes than is the unrolled-by-2 version
            //       below.   On the Cray t3d, the reverse is true, i.e., the
            //       unrolled-by-two version is some 10% faster.
            //       The unrolled-by-8 version below is significantly faster
            //       on the Cray t3d - overall speed of code is 1.5 times faster.
            //
            //not a perfectly nested rectangular loop, can't use collapse, inner loop is not equal length
            #pragma omp for private(sum)
            for (int j = 0; j < lastrow - firstrow + 1; j++)
            {
                sum = 0.0;
                //rowstr[j] and rowstr[j+1] brackets the slice of nonzeros belonging to row j in flat arrays a[] and colidx[]
                #pragma omp simd reduction(+:sum)
                for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
                {
                    //p[coldix[k]] is corresponding entry of vector p  for a[k].
                    //sum = sum + a[k] * p[colidx[k]];
                    sum += a[k] * p[colidx[k]];
                }
                //the loops gets dot product of row
                //q is 1-D vector of Ap
                q[j] = sum;
            }

            //---------------------------------------------------------------------
            // Obtain p.q
            //---------------------------------------------------------------------
            //d=p⊤(Ap)=p⊤q. a scalar of curvature 
            #pragma omp single
            {
                d = 0.0;
            }
            #pragma omp for reduction(+:d)
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                //d = d + p[j] * q[j];
                d += p[j] * q[j];
            }

            #pragma omp single
            {

                //---------------------------------------------------------------------
                // Obtain alpha = rho / (p.q)
                //---------------------------------------------------------------------
                // how far to move along rho
                alpha = rho / d;

                //---------------------------------------------------------------------
                // Save a temporary of rho
                //---------------------------------------------------------------------
                //need for beta = rho / rho0
                rho0 = rho;
            }
            //---------------------------------------------------------------------
            // Obtain z = z + alpha*p
            // and    r = r - alpha*q
            //---------------------------------------------------------------------
            #pragma omp single
            {
                rho = 0.0;
            }
            //writes to difference place, not a reduction
            #pragma omp  for
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                //update current answer and residual
                z[j] = z[j] + alpha * p[j]; // z ← z + α p
                r[j] = r[j] - alpha * q[j]; // r ← r − α (A p),  since q = A p
            }

            //---------------------------------------------------------------------
            // rho = r.r
            // Now, obtain the norm of r: First, sum squares of r elements locally...
            //---------------------------------------------------------------------
            #pragma omp for reduction(+:rho)
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                //rho = rho + r[j] * r[j]; // rho ← r·r  (new)
                rho += r[j] * r[j]; // rho ← r·r  (new)
            }

            //---------------------------------------------------------------------
            // Obtain beta:
            //---------------------------------------------------------------------
            #pragma omp single
            {
                beta = rho / rho0;
            }

            //---------------------------------------------------------------------
            // p = r + beta*p
            //---------------------------------------------------------------------
            #pragma omp for
            for (int j = 0; j < lastcol - firstcol + 1; j++)
            {
                //bulid new direction with a bit of old direction
                p[j] = r[j] + beta * p[j];
            }
        }
    } // end of do cgit=1,cgitmax

    //---------------------------------------------------------------------
    // Compute residual norm explicitly:  ||r|| = ||x - A.z||
    // First, form A.z
    // The partition submatrix-vector multiply
    //---------------------------------------------------------------------
    //forms Az is an 1-D vector, z is the answer after iterations
    sum = 0.0;
    #pragma omp parallel for private(d)
    for (int j = 0; j < lastrow - firstrow + 1; j++)
    {
        d = 0.0;
        #pragma omp simd reduction(+:d)
        for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
        {
            //d = d + a[k] * z[colidx[k]];
            d += a[k] * z[colidx[k]];
        }
        //r temporarily holds  Az (not the residual yet).
        r[j] = d;
    }

    //---------------------------------------------------------------------
    // At this point, r contains A.z
    //---------------------------------------------------------------------
    // compute final residual telling how good z is
    //!!! reduction keeps sum from data race but not d
    #pragma omp parallel for reduction(+:sum)
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        //d = x[j] - r[j];
        double diff = x[j] - r[j];
        //sum = sum + d * d;
        sum += diff * diff;
    }

    *rnorm = sqrt(sum);
}

//---------------------------------------------------------------------
// generate the test problem for benchmark 6
// makea generates a sparse matrix with a
// prescribed sparsity distribution
//
// parameter    type        usage
//
// input
//
// n            i           number of cols/rows of matrix
// nz           i           nonzeros as declared array size
// rcond        r*8         condition number
// shift        r*8         main diagonal shift
//
// output
//
// a            r*8         array for nonzeros
// colidx       i           col indices
// rowstr       i           row pointers
//
// workspace
//
// iv, arow, acol i
// aelt           r*8
//---------------------------------------------------------------------
void makea(int n,
           int nz,
           double a[],
           int colidx[],
           int rowstr[],
           int firstrow,
           int lastrow,
           int firstcol,
           int lastcol,
           int arow[],
           int acol[][NONZER + 1],
           double aelt[][NONZER + 1],
           int iv[])
{
    int nzv, nn1;
    int ivc[NONZER + 1];
    double vc[NONZER + 1];

    //---------------------------------------------------------------------
    // nonzer is approximately  (int(sqrt(nnza /n)));
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // nn1 is the smallest power of two not less than n
    //---------------------------------------------------------------------
    nn1 = 1;
    do
    {
        nn1 = 2 * nn1;
    } while (nn1 < n);

    //---------------------------------------------------------------------
    // Generate nonzero positions and save for the use in sparse.
    //---------------------------------------------------------------------
    for (int iouter = 0; iouter < n; iouter++)
    {
        nzv = NONZER;
        sprnvc(n, nzv, nn1, vc, ivc);
        vecset(n, vc, ivc, &nzv, iouter + 1, 0.5);
        arow[iouter] = nzv;

        for (int ivelt = 0; ivelt < nzv; ivelt++)
        {
            acol[iouter][ivelt] = ivc[ivelt] - 1;
            aelt[iouter][ivelt] = vc[ivelt];
        }
    }

    //---------------------------------------------------------------------
    // ... make the sparse matrix from list of elements with duplicates
    //     (iv is used as  workspace)
    //---------------------------------------------------------------------
    sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, aelt, firstrow, lastrow, iv, RCOND, SHIFT);
}

//---------------------------------------------------------------------
// rows range from firstrow to lastrow
// the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
//---------------------------------------------------------------------
void sparse(double a[],
            int colidx[],
            int rowstr[],
            int n,
            int nz,
            int nozer,
            const int arow[],
            int acol[][NONZER + 1],
            double aelt[][NONZER + 1],
            int firstrow,
            int lastrow,
            int nzloc[],
            double rcond,
            double shift)
{
    //---------------------------------------------------
    // generate a sparse matrix from a list of
    // [col, row, element] tri
    //---------------------------------------------------
    int nza, nzrow, jcol;
    double size, scale, ratio, va;
    bool cont40;

    //---------------------------------------------------------------------
    // how many rows of result
    //---------------------------------------------------------------------
    const int nrows = lastrow - firstrow + 1;

    //---------------------------------------------------------------------
    // ...count the number of triples in each row
    //---------------------------------------------------------------------
    for (int j = 0; j < nrows + 1; j++)
    {
        rowstr[j] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (nza = 0; nza < arow[i]; nza++)
        {
            int j = acol[i][nza] + 1;
            rowstr[j] = rowstr[j] + arow[i];
        }
    }

    rowstr[0] = 0;
    for (int j = 1; j < nrows + 1; j++)
    {
        rowstr[j] = rowstr[j] + rowstr[j - 1];
    }
    nza = rowstr[nrows] - 1;

    //---------------------------------------------------------------------
    // ... rowstr(j) now is the location of the first nonzero
    //     of row j of a
    //---------------------------------------------------------------------
    if (nza > nz)
    {
        printf("Space for matrix elements exceeded in sparse\n");
        printf("nza, nzmax = %d, %d\n", nza, nz);
        exit(EXIT_FAILURE);
    }

    //---------------------------------------------------------------------
    // ... preload data pages
    //---------------------------------------------------------------------
    for (int j = 0; j < nrows; j++)
    {
        for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
        {
            a[k] = 0.0;
            colidx[k] = -1;
        }
        nzloc[j] = 0;
    }

    //---------------------------------------------------------------------
    // ... generate actual values by summing duplicates
    //---------------------------------------------------------------------
    size = 1.0;
    ratio = pow(rcond, (1.0 / (double)(n)));

    for (int i = 0; i < n; i++)
    {
        for (nza = 0; nza < arow[i]; nza++)
        {
            int j = acol[i][nza];

            scale = size * aelt[i][nza];
            for (nzrow = 0; nzrow < arow[i]; nzrow++)
            {
                int jcol = acol[i][nzrow];
                va = aelt[i][nzrow] * scale;

                //--------------------------------------------------------------------
                // ... add the identity * rcond to the generated matrix to bound
                //     the smallest eigenvalue from below by rcond
                //--------------------------------------------------------------------
                if (jcol == j && j == i)
                {
                    va = va + rcond - shift;
                }

                cont40 = false;
                int k = rowstr[j];
                for (; k < rowstr[j + 1]; k++)
                {
                    if (colidx[k] > jcol)
                    {
                        //----------------------------------------------------------------
                        // ... insert colidx here orderly
                        //----------------------------------------------------------------
                        for (int kk = rowstr[j + 1] - 2; kk >= k; kk--)
                        {
                            if (colidx[kk] > -1)
                            {
                                a[kk + 1] = a[kk];
                                colidx[kk + 1] = colidx[kk];
                            }
                        }
                        colidx[k] = jcol;
                        a[k] = 0.0;
                        cont40 = true;
                        break;
                    }
                    if (colidx[k] == -1)
                    {
                        colidx[k] = jcol;
                        cont40 = true;
                        break;
                    }
                    if (colidx[k] == jcol)
                    {
                        //--------------------------------------------------------------
                        // ... mark the duplicated entry
                        //--------------------------------------------------------------
                        nzloc[j] = nzloc[j] + 1;
                        cont40 = true;
                        break;
                    }
                }
                if (cont40 == false)
                {
                    printf("internal error in sparse: i=%d\n", i);
                    exit(EXIT_FAILURE);
                }
                a[k] = a[k] + va;
            }
        }
        size = size * ratio;
    }

    //---------------------------------------------------------------------
    // ... remove empty entries and generate final results
    //---------------------------------------------------------------------
    for (int j = 1; j < nrows; j++)
    {
        nzloc[j] = nzloc[j] + nzloc[j - 1];
    }

    for (int j = 0; j < nrows; j++)
    {
        int j1 = 0;
        if (j > 0)
        {
            j1 = rowstr[j] - nzloc[j - 1];
        }
        else
        {
            j1 = 0;
        }
        int j2 = rowstr[j + 1] - nzloc[j];
        nza = rowstr[j];
        for (int k = j1; k < j2; k++)
        {
            a[k] = a[nza];
            colidx[k] = colidx[nza];
            nza = nza + 1;
        }
    }
    for (int j = 1; j < nrows + 1; j++)
    {
        rowstr[j] = rowstr[j] - nzloc[j - 1];
    }
}

//---------------------------------------------------------------------
// generate a sparse n-vector (v, iv)
// having nzv nonzeros
//
// mark(i) is set to 1 if position i is nonzero.
// mark is all zero on entry and is reset to all zero before exit
// this corrects a performance bug found by John G. Lewis, caused by
// reinitialization of mark on every one of the n calls to sprnvc
//---------------------------------------------------------------------
void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
    double vecelt, vecloc;

    int nzv = 0;

    while (nzv < nz)
    {
        vecelt = randlc(&tran, amult);

        //---------------------------------------------------------------------
        // generate an integer between 1 and n in a portable manner
        //---------------------------------------------------------------------
        vecloc = randlc(&tran, amult);
        int i = icnvrt(vecloc, nn1) + 1;
        if (i > n)
            continue;

        //---------------------------------------------------------------------
        // was this integer generated already?
        //---------------------------------------------------------------------
        bool was_gen = false;
        for (int ii = 0; ii < nzv; ii++)
        {
            if (iv[ii] == i)
            {
                was_gen = true;
                break;
            }
        }
        if (was_gen)
            continue;
        v[nzv] = vecelt;
        iv[nzv] = i;
        nzv = nzv + 1;
    }
}

//---------------------------------------------------------------------
// scale a double precision number x in (0,1) by a power of 2 and chop it
//---------------------------------------------------------------------
int icnvrt(double x, int ipwr2)
{
    return (int)(ipwr2 * x);
}

//---------------------------------------------------------------------
// set ith element of sparse vector (v, iv) with
// nzv nonzeros to val
//---------------------------------------------------------------------
void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
    bool set = false;
    for (int k = 0; k < *nzv; k++)
    {
        if (iv[k] == i)
        {
            v[k] = val;
            set = true;
        }
    }
    if (set == false)
    {
        v[*nzv] = val;
        iv[*nzv] = i;
        *nzv = *nzv + 1;
    }
}

void init(double *zeta)
{
    firstrow = 0;
    lastrow = NA - 1;
    firstcol = 0;
    lastcol = NA - 1;

    naa = NA;
    nzz = NZ;

    //---------------------------------------------------------------------
    // Inialize random number generator
    //---------------------------------------------------------------------
    tran = 314159265.0;
    amult = 1220703125.0;
    *zeta = randlc(&tran, amult);

    //---------------------------------------------------------------------
    //
    //---------------------------------------------------------------------
    makea(naa, nzz, a, colidx, rowstr, firstrow, lastrow, firstcol, lastcol, arow,
          (int(*)[NONZER + 1]) acol, (double(*)[NONZER + 1]) aelt, iv);

    //---------------------------------------------------------------------
    // Note: as a result of the above call to makea:
    //      values of j used in indexing rowstr go from 0 --> lastrow-firstrow
    //      values of colidx which are col indexes go from firstcol --> lastcol
    //      So:
    //      Shift the col index vals from actual (firstcol --> lastcol )
    //      to local, i.e., (0 --> lastcol-firstcol)
    //---------------------------------------------------------------------
    for (int j = 0; j < lastrow - firstrow + 1; j++)
    {
        for (int k = rowstr[j]; k < rowstr[j + 1]; k++)
        {
            colidx[k] = colidx[k] - firstcol;
        }
    }

    //---------------------------------------------------------------------
    // set starting vector to (1, 1, .... 1)
    //---------------------------------------------------------------------
    for (int i = 0; i < NA + 1; i++)
    {
        x[i] = 1.0;
    }
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        q[j] = 0.0;
        z[j] = 0.0;
        r[j] = 0.0;
        p[j] = 0.0;
    }
}

void iterate(double *zeta, const int *it)
{
    double rnorm;
    double norm_temp1, norm_temp2;

    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    norm_temp1 = 0.0;
    norm_temp2 = 0.0;
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        norm_temp1 = norm_temp1 + x[j] * z[j];
        norm_temp2 = norm_temp2 + z[j] * z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    *zeta = SHIFT + 1.0 / norm_temp1;
    if (*it == 1)
        printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", *it, rnorm, *zeta);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    for (int j = 0; j < lastcol - firstcol + 1; j++)
    {
        x[j] = norm_temp2 * z[j];
    }
}

