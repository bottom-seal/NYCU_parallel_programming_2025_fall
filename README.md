# NYCU Parallel Programming 2025 Fall

Implementations for all labs in the course.

## Scores

| Item | Score |
|-----|------|
| hw1 | 100 |
| hw2 | 100 |
| hw3 | 92.865 |
| hw4 | 93 |
| hw5 | 98.72819512 |
| hw6 | 99.93650794 |
| **Total** | **94.94107002** |

---

## Implementation Notes

Only techniques beyond the basic course material are listed here.

---

## HW1 – SIMD Programming

- Masking to handle conditional logic (`x < 0` vs `x >= 0`)
- Boundary masking when `N` is not a multiple of `VECTOR_WIDTH`
- Tree reduction (pairwise reduction) for vector summation

---

## HW2 – SIMD + Multithreading

### Part 1 – Monte Carlo π

- **Xoshiro256PlusSIMD** library for random number generation
- **AVX intrinsics**
  - `_mm256_mul_ps`
  - `_mm256_fmadd_ps`

### Part 2 – Mandelbrot

- Changed **block partitioning → cyclic partitioning** to improve load balance
- Used **SIMD inside `mandel()`** for full score

---

## HW3 – Parallel BFS (OpenMP)

- Standard **OpenMP parallelization**
- Implemented both required parts (not fully optimized)

---

## HW4 – MPI Matrix Multiplication

### Part 1

- Basic implementation is sufficient for full score

### Part 2

Optimizations used:

- Store **matrix B in column-major layout**
- **Loop tiling (cache blocking)** for matrix multiplication
- **Row-block partitioning** across MPI processes
- Used **`MPI_Gatherv`** to collect results

---

## HW5 – CUDA Mandelbrot

- **Manual loop unrolling** in `mandel()`
- **Zero-copy memory**
  - mapped pinned host memory

---

## HW6 – OpenCL Convolution
You might want to just check the code<br>
Optimizations used:

- **Fused multiply-add** (`mad()`)
- **Tiled convolution**
- **Strided cooperative loading** into local memory
- Each **work-item computes 4 output pixels** (thread coarsening)
- Use **local memory (shared memory)** as a cache
- Smaller **work-group size** works better since each thread computes multiple pixels
- Pass **input tile size** as a kernel parameter for indexing

## FINAL
- https://github.com/bottom-seal/Parallel_A_star
