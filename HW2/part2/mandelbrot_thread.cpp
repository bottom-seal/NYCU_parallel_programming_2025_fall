#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>

using SteadyClock = std::chrono::steady_clock;

namespace
{
// can't use AVX2 somehow, vectorize to 8 so -O3 flag automatically do it
inline void mandel8_precise(const float *c_re, const float *c_im,
                            int max_iter, int *result)
{
    // do 8 at once for speed
    float zr[8], zi[8];
    int iter[8];
    for (int k = 0; k < 8; ++k)
    {
        zr[k] = c_re[k];
        zi[k] = c_im[k];
        iter[k] = 0;
    }

    for (int i = 0; i < max_iter; ++i)
    {
        //bool tells if all 8 escaped, like a mask
        bool active = false;
        //checks 8 pixels
        for (int k = 0; k < 8; ++k)
        {
            float z2r = zr[k] * zr[k];
            float z2i = zi[k] * zi[k];
            if (z2r + z2i > 4.f)
                continue;
            // if all pixel escaped, no one will set active to true, loop will break
            active = true;
            iter[k]++;
            float new_re = z2r - z2i + c_re[k];
            float new_im = 2.f * zr[k] * zi[k] + c_im[k];
            zr[k] = new_re;
            zi[k] = new_im;
        }
        if (!active)
            break;
    }
    for (int k = 0; k < 8; ++k)
        result[k] = iter[k];
}
//for last few that cannot be a team of 8, use scalar
inline int mandel_scalar(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i = 0;
    for (; i < count; ++i)
    {
        float z2_re = z_re * z_re;
        float z2_im = z_im * z_im;
        if (z2_re + z2_im > 4.f)
            break;
        float new_re = z2_re - z2_im + c_re;
        float new_im = 2.f * z_re * z_im + c_im;
        z_re = new_re;
        z_im = new_im;
    }
    return i;
}

void mandelbrot_serial_precise(float x0, float y0, float x1, float y1,
                               int width, int height,
                               int start_row, int step,
                               int max_iterations,
                               int *__restrict output)
{
    const float dx = (x1 - x0) / (float)width;
    const float dy = (y1 - y0) / (float)height;

    float xbuf[8], ybuf[8];

    for (int j = start_row; j < height; j += step)
    {
        const float y = y0 + (float)j * dy;
        const int row_offset = j * width;

        int i = 0;
        //The part we timed up, now we can do 8 in one loop
        for (; i + 7 < width; i += 8)
        {
            for (int k = 0; k < 8; ++k)
            {
                xbuf[k] = x0 + (float)(i + k) * dx;
                ybuf[k] = y;
            }

            int iters[8];
            mandel8_precise(xbuf, ybuf, max_iterations, iters);
            for (int k = 0; k < 8; ++k)
                output[row_offset + i + k] = iters[k];
        }
        // for the leftover <8 part
        for (; i < width; ++i)
        {
            float x = x0 + (float)i * dx;
            output[row_offset + i] = mandel_scalar(x, y, max_iterations);
        }
    }
}
} // namespace

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
    double *timing_ms;
};

void worker_thread_start(WorkerArgs *const args)
{
    auto t0 = SteadyClock::now();

    mandelbrot_serial_precise(args->x0, args->y0, args->x1, args->y1,
                              args->width, args->height,
                              args->threadId, args->numThreads,
                              args->maxIterations,
                              args->output);

    auto t1 = SteadyClock::now();
    args->timing_ms[args->threadId] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void mandelbrot_thread(int num_threads,
                       float x0, float y0,
                       float x1, float y1,
                       int width, int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;
    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args{};
    std::vector<double> timing(num_threads, 0.0);

    for (int i = 0; i < num_threads; i++)
    {
        args[i] = {x0, x1, y0, y1,
                   (unsigned)width, (unsigned)height,
                   max_iterations, output,
                   i, num_threads, timing.data()};
    }

    for (int i = 1; i < num_threads; i++)
        workers[i] = std::thread(worker_thread_start, &args[i]);

    worker_thread_start(&args[0]);

    for (int i = 1; i < num_threads; i++)
        workers[i].join();

    for (int t = 0; t < num_threads; ++t)
        printf("[thread %d] %.3f ms\n", t, timing[t]);
}

