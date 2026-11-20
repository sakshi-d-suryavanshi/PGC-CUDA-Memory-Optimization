# Experiment 6: Memory Optimization in CUDA

**Subject:** Memory Systems and Synchronization  
**Experiment:** Experiment 6 – Memory Optimization in CUDA  
**Language:** CUDA C/C++  
**Platform:** Ubuntu on WSL  
**GPU:** NVIDIA RTX 4500 Ada Generation  
**Compute Capability:** 8.9  
**CUDA Toolkit:** 12.4  

---

## 1. Aim

To study how global-memory access patterns affect CUDA kernel performance by comparing uncoalesced and coalesced matrix transpose operations, implementing naive matrix multiplication, and measuring their execution times on an NVIDIA GPU.

## 2. Objectives

- Understand coalesced and uncoalesced global-memory access.
- Implement a matrix transpose using CUDA.
- Compare uncoalesced and coalesced memory access.
- Use shared memory to improve memory-access efficiency.
- Implement naive matrix multiplication using CUDA.
- Measure GPU execution time using CUDA events.
- Calculate effective memory bandwidth and floating-point performance.
- Verify GPU results against CPU reference calculations.

## 3. Hardware and Software Requirements

### Hardware

- NVIDIA GPU
- Sufficient GPU memory for matrix operations

### Software

- Ubuntu on Windows Subsystem for Linux (WSL)
- NVIDIA GPU driver
- CUDA Toolkit
- NVIDIA CUDA Compiler (`nvcc`)
- C/C++ compiler

### System Configuration Used

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation |
| Compute Capability | 8.9 |
| CUDA Toolkit | 12.4 |
| CUDA Compiler | NVCC |
| Programming Language | CUDA C/C++ |
| Operating Environment | Ubuntu on WSL |

## 4. Theory

### 4.1 Memory Coalescing

Memory coalescing is a technique used by GPUs to combine memory requests from multiple threads into fewer memory transactions.

A group of 32 CUDA threads is called a **warp**.

When threads in a warp access consecutive global-memory addresses, the GPU can combine these requests efficiently.

This is called coalesced memory access.

When threads access scattered or widely separated memory locations, multiple memory transactions may be required. This is called uncoalesced memory access.

| Coalesced Access | Uncoalesced Access |
|---|---|
| Threads access neighbouring addresses. | Threads access scattered addresses. |
| Requires fewer memory transactions. | Requires more memory transactions. |
| Better memory efficiency. | Lower memory efficiency. |
| Higher effective bandwidth. | Lower effective bandwidth. |

### 4.2 Matrix Transpose

Matrix transpose converts rows into columns and columns into rows.

For an input matrix:

`A[i][j]`

The transposed output is:

`B[j][i]`

The transpose operation can be represented as:

`B[j][i] = A[i][j]`

Two CUDA implementations are used.

#### A. Uncoalesced Matrix Transpose

- Each CUDA thread copies one matrix element.
- Consecutive threads read consecutive elements.
- However, their output writes are separated by a stride of N elements.
- This causes inefficient global-memory transactions.

#### B. Coalesced Matrix Transpose

- Uses shared memory to temporarily store matrix elements.
- Loads a 32 × 32 tile into shared memory.
- Uses `__syncthreads()` to synchronize threads.
- Writes the elements in transposed order.
- Uses `tile[TILE][TILE+1]` to reduce shared-memory bank conflicts.

The coalesced implementation improves memory-access efficiency by making both global-memory reads and writes contiguous.

### 4.3 Naive Matrix Multiplication

Matrix multiplication calculates the product of two matrices.

For two matrices A and B:

`C = A × B`

Each output element is calculated as:

`C[i][j] = Σ A[i][k] × B[k][j]`

A single CUDA thread calculates one element of the output matrix.

Each thread performs:
- N iterations.
- Two global-memory reads per iteration.
- One multiplication and one addition per iteration.

The naive implementation does not use shared-memory tiling to reuse matrix elements. Therefore, the same data may be loaded repeatedly from global memory.

## 5. CUDA Implementation

The experiment is implemented using three CUDA kernels.

| Kernel | Purpose | Configuration |
|---|---|---|
| `transposeNaive` | Performs uncoalesced matrix transpose | 2D CUDA grid and block |
| `transposeCoalesced` | Performs coalesced matrix transpose using shared memory | 32 × 32 tile |
| `matMulNaive` | Performs naive matrix multiplication | 16 × 16 block |

### Important CUDA Concepts Used

**1. Global Memory**

Global memory is the main GPU memory accessible by all CUDA threads. It provides large storage capacity but has higher latency than shared memory.

**2. Shared Memory**

Shared memory is a fast memory space shared by threads within a CUDA block. It helps reduce repeated global-memory accesses.

**3. `__syncthreads()`**

This function synchronizes threads within a CUDA block. It ensures that threads finish loading data into shared memory before other threads read that data.

**4. CUDA Events**

CUDA events are used to measure GPU execution time accurately.

The program performs warm-up launches and repeated kernel executions to obtain more reliable timing measurements.

**5. Shared-Memory Padding**

The shared-memory tile is declared as:

`tile[TILE][TILE+1]`

The additional column helps avoid shared-memory bank conflicts during transposed access.

## 6. Project Structure

```text
Exp6_CUDA_Memory_Optimization/
│
├── exp6.cu
│   └── Complete CUDA implementation
│
├── exp6
│   └── Compiled executable
│
└── README.md
    └── Experiment documentation
```

## 7. Compilation and Execution

### Step 1: Open the project directory

```bash
cd ~/Exp6_CUDA_Memory_Optimization
```

### Step 2: Verify NVIDIA GPU

```bash
nvidia-smi
```

### Step 3: Verify CUDA compiler

```bash
nvcc --version
```

### Step 4: Compile the CUDA program

```bash
nvcc -O2 -arch=sm_89 -o exp6 exp6.cu
```

Here:

- `nvcc` invokes the NVIDIA CUDA compiler.
- `-O2` enables compiler optimization.
- `-arch=sm_89` targets the RTX 4500 Ada GPU architecture.
- `-o exp6` specifies the executable name.

### Step 5: Execute the program

```bash
./exp6
```

The default matrix sizes are:

- Matrix transpose: 4096 × 4096
- Matrix multiplication: 1024 × 1024

### Step 6: Repeat the experiment

Run the program at least three times:

```bash
./exp6
./exp6
./exp6
```

Record the execution times and performance measurements from each run.

## 8. Experimental Results

The experiment was executed on an NVIDIA RTX 4500 Ada Generation GPU with compute capability 8.9.

### 8.1 Results from Three Runs

| Measurement | Run 1 | Run 2 | Run 3 | Average |
|---|---:|---:|---:|---:|
| Uncoalesced transpose (ms) | 0.712 | 0.720 | 0.718 | **0.717** |
| Coalesced transpose (ms) | 0.369 | 0.369 | 0.369 | **0.369** |
| Naive matrix multiplication (ms) | 0.896 | 0.892 | 0.892 | **0.893** |
| Uncoalesced bandwidth (GB/s) | 188.6 | 186.5 | 186.9 | **187.3** |
| Coalesced bandwidth (GB/s) | 364.1 | 363.8 | 363.9 | **363.9** |
| Transpose speedup | 1.93× | 1.95× | 1.95× | **1.94×** |
| Matrix multiplication (GFLOPS) | 2397.9 | 2407.8 | 2408.4 | **2404.7** |

### 8.2 Observation Table

| Kernel | Matrix Size | Average Time | Speedup | Performance | Result |
|---|---:|---:|---:|---:|---|
| Uncoalesced transpose | 4096 | 0.717 ms | 1.0× | 187.3 GB/s | PASS |
| Coalesced transpose | 4096 | 0.369 ms | 1.94× | 363.9 GB/s | PASS |
| Naive matrix multiplication | 1024 | 0.893 ms | - | 2404.7 GFLOPS | PASS |

**Average speedup of coalesced transpose: 1.94×**

### 8.3 Verification Results

All verification checks passed successfully.

```text
Uncoalesced transpose : PASS
Coalesced transpose   : PASS
Naive matrix multiplication : PASS

All verification checks passed successfully!
```

## 9. Performance Calculation

### 9.1 Effective Memory Bandwidth

The effective bandwidth for matrix transpose is calculated using:

`Bandwidth = (2 × N × N × 4) / Time`

The factor 2 represents one read and one write for each matrix element.

The factor 4 represents the size of a float in bytes.

Time is converted into seconds, and bandwidth is expressed in GB/s.

### 9.2 Matrix Multiplication Performance

The floating-point performance is calculated using:

`GFLOPS = (2 × N³) / (Time × 10⁶)`

Here:

- N is the matrix dimension.
- 2 represents one multiplication and one addition.
- Time is measured in milliseconds.

### 9.3 Speedup Calculation

`Speedup = Uncoalesced Execution Time / Coalesced Execution Time`

Using the average measurements:

`Speedup = 0.717 / 0.369`

`Speedup ≈ 1.94×`

This means the coalesced transpose is approximately 1.94 times faster than the uncoalesced transpose in this experiment.

## 10. Analysis and Discussion

### 10.1 Why Does Coalesced Transpose Perform Better?

In uncoalesced transpose, consecutive threads write to memory locations that are separated by a large stride.

This results in inefficient global-memory transactions.

In coalesced transpose, shared memory is used to reorganize the matrix tile. Both global-memory reads and writes become contiguous.

Therefore, memory-access efficiency and effective bandwidth improve.

### 10.2 Why Is Shared-Memory Padding Required?

The coalesced transpose uses:

`tile[TILE][TILE+1]`

Without padding, threads accessing a column of the shared-memory tile may experience bank conflicts.

Adding one extra column changes the memory layout and helps distribute accesses across shared-memory banks.

### 10.3 Why Is Naive Matrix Multiplication Less Optimized?

In naive matrix multiplication, each thread independently calculates one output element.

The same input matrix elements are loaded repeatedly from global memory.

There is no shared-memory tiling to reuse the loaded data.

A tiled matrix multiplication implementation could reduce repeated global-memory accesses and improve performance.

### 10.4 Why Are CUDA Events Used?

CUDA kernel launches are asynchronous with respect to the CPU.

A CPU timer around a kernel launch may measure launch overhead rather than actual GPU execution time.

CUDA events measure elapsed time on the GPU and provide more appropriate kernel timing.

## 11. Conclusion

The experiment successfully demonstrated the effect of memory-access patterns on CUDA performance.

Both uncoalesced and coalesced matrix transpose implementations produced correct results.

The coalesced transpose achieved an average execution time of 0.369 ms, compared with 0.717 ms for the uncoalesced transpose.

This resulted in an average speedup of approximately 1.94×.

The naive matrix multiplication also produced correct results, achieving an average performance of approximately 2404.7 GFLOPS.

Thus, the experiment demonstrated that efficient global-memory access, shared-memory usage, and proper synchronization play important roles in improving GPU performance.

**Final Result:** The CUDA memory optimization experiment was successfully implemented and executed. All verification checks passed.
.

---

**Experiment Status: Successfully Completed**

**GPU Verification: All PASS**

**Number of Performance Runs: 3**
