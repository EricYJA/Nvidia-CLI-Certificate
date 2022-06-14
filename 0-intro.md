## CUDA Intro

This is a brief CUDA tutorial and my personal CUDA study note. Please let me know if
I made any mistake. 



### CPU & GPU

**CPU**

- Latency-oriented design

**GPU**

- throughput-oriented design

However, if the GPU is“over-subscribed” with threads, that is, it runs significantly more threads than physical cores, it can hide these latencies by quickly switching execution between those threads.



*How to understand this sentence?*



### Background of GPU

As the parallelism of GPU began to take over many task execution, the limitation of GPU began to emerge. Here we introduce the three walls as defined by D. Patterson at UC Berkeley

- Power Wall:  Cooling expanses not economized by additional performance
- Memory Wall: Multiple fast cores are bottlenecked by slow main memory
- ILP(Instruction Level Parallelism): There is only so much prediction and pipelining you can do



### CUDA (Compute Unified Device Architecture)





### Execution Scheme

 CUDA uses a asynchronized scheme between CPU and GPU. That is, briefly, we would first initialize the task and distribute them on the CPU and GPU and let them run separately. Whenever we want to sync or verify the computation result. we should manually let them synchronize. 



### Key words

Code Example: `0-intro.cu`

```c++
// accessable by GPU and CPU, must return void
__global__
    
// kernel function for exe on GPU, specify what to use, etc: <<<block num, thread num per block>>> 
GPUFunction<<<1, 1>>>();

// let CPU and GPU wait for each other and sync data
cudaDeviceSynchronize();

// build-in param for block ID and thread ID
blockIdx.x;
threadIdx.x;
```

 

