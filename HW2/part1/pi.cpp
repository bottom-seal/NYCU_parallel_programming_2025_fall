// pi_multi_thread.c
#define __AVX2_AVAILABLE__ 
#include <iostream>
#include <string>
#include <cstdlib> 
#include <pthread.h>
#include <cstdint>
#include <immintrin.h>

#include "include/SIMDInstructionSet.h"
#include "include/Xoshiro256Plus.h"
typedef SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> Xoshiro256PlusAVX2;
using namespace SEFUtility::RNG;
//#define NUMTHRDS 4
//#define MAGNIFICATION 1e9

struct Arg {
    int thread_id;
    int start;
    int end;
    //long long* land_num;
    //no longer need mutex to save time
    long long* out;
}; // 傳入 thread 的參數型別

pthread_mutex_t mutexsum;
// 每個 thread 要做的任務
// 
void *count_pi(void *arg)
{

   Arg *data = (Arg *)arg;
   int thread_id = data->thread_id;
   int start = data->start;
   int end = data->end;
   //long long *land_num = data->land_num;
   long long *out = data->out; 
   // 將原本的 PI 算法切成好幾份
   // modified to dart tossing task
   const uint64_t base_seed = 0xC0FFEE123456789ULL;
    Xoshiro256Plus<SIMDInstructionSet::AVX2> rng(
    base_seed ^ (uint64_t)thread_id * 0x9E3779B97F4A7C15ULL);

    long long local_land_num = 0;

    int n = end - start;
    int n8 = n / 8;   
    //int rem = n % 8;   // need to deal with few darts that cannot be grouped

    const __m256 kScale = _mm256_set1_ps(4.6566128730773926e-10f); // 1 / 2^31
    const __m256 onef   = _mm256_set1_ps(1.0f);
    // --- 8 tosses per iteration (16 doubles) ---
    // 8 tosses per iteration: need 16 doubles total (x0..x7, y0..y7)
    for (int i = 0; i < n8; ++i) {
        __m256 xf  = _mm256_mul_ps(_mm256_cvtepi32_ps(rng.next4().operator __m256i()), kScale);
        __m256 yf  = _mm256_mul_ps(_mm256_cvtepi32_ps(rng.next4().operator __m256i()), kScale);
        // dist2 = x*x + y*y (use FMA)
        __m256 d0 = _mm256_fmadd_ps(xf,  xf,  _mm256_mul_ps(yf,  yf));

        // (dist2 <= 1) → mask → popcount
        int m0 = _mm256_movemask_ps(_mm256_cmp_ps(d0, onef, _CMP_LE_OS));

        local_land_num += __builtin_popcount(m0);
    }

    // ---- remainder (scalar) ----
    //for (int i = 0; i < rem; ++i) {
    //    double x = rng.dnext(-1.0, 1.0);
    //    double y = rng.dnext(-1.0, 1.0);
    //    if (x * x + y * y <= 1.0)
    //        ++local_land_num;
    //}
   // **** 關鍵區域 ****
   // 一次只允許一個 thread 存取
   //pthread_mutex_lock(&mutexsum);
   // 將部分的 PI 加進最後的 PI
   //*land_num += local_land_num;
   //pthread_mutex_unlock(&mutexsum);
   // *****************
    *out = local_land_num;
   //printf("Thread %d did %d to %d:  local land=%lld global land num=%lld\n", thread_id, start,
   //       end, local_land_num, *land_num);

   pthread_exit((void *)0);
}

int main(int argc, char** argv)
{
    //new args each time program runs
    int        num_threads = 1;
    long long  total_steps = 1;
    if (argc >= 2) num_threads = std::stoi(argv[1]);  // parse int
    if (argc >= 3) total_steps = std::stoll(argv[2]); // parse long long

    //dynamic allocate since se take argument each time it runs
    pthread_t* callThd = new pthread_t[num_threads];// 宣告建立 pthread
    Arg* arg = new Arg[num_threads]; 
    // 初始化互斥鎖
    //pthread_mutex_init(&mutexsum, NULL);

    //each thread has its own slot 
    //alignas(64) long long *results = new long long[num_threads]();

    //eliminate false sharing
    struct alignas(64) Slot { long long v; char pad[64 - sizeof(long long)]; };
    Slot* results = (Slot*) aligned_alloc(64, sizeof(Slot)*num_threads);

    // 設定 pthread 性質是要能 join
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // 每個 thread 都可以存取的 PI
    // 因為不同 thread 都要能存取，故用指標
    //long long *land_num = (long long*)malloc(sizeof(*land_num));
    //*land_num = 0;

    int part = total_steps / num_threads;

    for (int i = 0; i < num_threads; i++)
    {
        // 設定傳入參數
        arg[i].thread_id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i + 1);
        //arg[i].land_num = land_num; // PI 的指標，所有 thread 共用
        arg[i].out = &results[i].v;
        // 建立一個 thread，執行 count_pi 任務，傳入 arg[i] 指標參數
        pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
    }

    // 回收性質設定
    pthread_attr_destroy(&attr);

    void *status;
    for (int i = 0; i < num_threads; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(callThd[i], &status);
    }

    // 所有 thread 執行完畢，印出 PI
    //double pi = 4.0 * (*land_num) / static_cast<double>(total_steps);
    long long total_num = 0;
    for (int i = 0; i < num_threads; ++i) 
        total_num += results[i].v;
    double pi = 4.0 * total_num / static_cast<double>(total_steps);
    printf("%.10lf\n", pi);

    // 回收互斥鎖
    pthread_mutex_destroy(&mutexsum);
    // 離開
    pthread_exit(NULL);
}
