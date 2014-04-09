// -*- C++ -*-
#include <cstdio>

#include <cuda_runtime.h>

#include "Main1_cuda.cuh"

//since we can't really dynamically size this array,
//let's leave its size at the default polynomial order
__constant__ float constant_c[10];


__global__
void
cudaSum_atomic_kernel(const float* const inputs,
                                     unsigned int numberOfInputs,
                                     const float* const c,
                                     unsigned int polynomialOrder,
                                     float* output) {

    float partialSum = 0.0;

    unsigned int inputIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int i;
    float p, r;

    if (polynomialOrder == 0)
       return;

    while (inputIndex < numberOfInputs) {
      partialSum += c[0];

      p = 1.0;
      r = inputs[inputIndex];
      for (i = 1; i < polynomialOrder; ++i) {
        partialSum += c[i] * (p *= r);
      }

      inputIndex += blockDim.x * gridDim.x;
    }

    atomicAdd(output, partialSum);
}

__global__
void
cudaSum_linear_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs, 
                                  const float* const c,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    extern __shared__ float partial_outputs[];

    // Initialize shared memory to 0.0
    // if (threadIdx.x == 0) {
    //   for (unsigned int i = 0; i < blockDim.x; ++i)
    //     partial_outputs[i] = 0.0;
    // }

    // syncthreads();

    unsigned int inputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float * const pout = partial_outputs + threadIdx.x;

    *pout = 0;

    if (polynomialOrder == 0)
       return;

    while (inputIndex < numberOfInputs) {
      *pout += c[0];

      float p = 1.0;
      float r = inputs[inputIndex];
      
      for (unsigned int i = 1; i < polynomialOrder; ++i) {
        *pout += c[i] * (p *= r);
      }

      inputIndex += blockDim.x * gridDim.x;
    }

    syncthreads();

    // Accumulate results from shared memory
    if (threadIdx.x == 0) {
      float partialSum = 0.0;
      for (unsigned int i = 0; i < blockDim.x; ++i)
        partialSum += partial_outputs[i];
      atomicAdd(output, partialSum);
    }
}


/* Used in Assignment 2. Coming soon! */
__global__
void
cudaSum_divtree_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs, 
                                  const float* const c,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    

}

/* Used in Assignment 2. Coming soon! */
__global__
void
cudaSum_nondivtree_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs, 
                                  const float* const c,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    

}

/* Used in Assignment 2. Coming soon! */
__global__
void
cudaSum_constmem_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    

}




void
cudaSumPolynomials(const float* const input,
                            const size_t numberOfInputs,
                            const float* const c,
                            const size_t polynomialOrder,
                            const Style style,
                            const unsigned int maxBlocks,
                            float * const output) {


    //Input values (your "r" values) go here on the GPU
    float *dev_input;
    
    //Your polynomial coefficients go here (GPU)
    float *dev_c;
    
    //Your output will go here (GPU)
    float *dev_output;
    const float float_zero = 0.0f;

    // Allocate memory for GPU to hold inputs
    cudaMalloc((void **) &dev_input,  numberOfInputs  * sizeof(float));
    cudaMemcpy(dev_input, input,  numberOfInputs  * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dev_c,      polynomialOrder * sizeof(float));
    cudaMemcpy(dev_c,     c,      polynomialOrder * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for GPU to hold output
    cudaMalloc((void **) &dev_output, sizeof(float));
    cudaMemcpy(dev_output, &float_zero, sizeof(float), cudaMemcpyHostToDevice);
    
    const unsigned int threadsPerBlock = 512;
    const unsigned int blocks 
                = min((float)maxBlocks, 
                        ceil(numberOfInputs/(float)threadsPerBlock));
    

    if (style == mutex) {
        cudaSum_atomic_kernel<<<blocks, threadsPerBlock>>>(
                dev_input, numberOfInputs, dev_c, polynomialOrder, dev_output);
    } else if (style == linear) {
        cudaSum_linear_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                dev_c, polynomialOrder, dev_output);
    } else if (style == divtree) {
        cudaSum_divtree_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                dev_c, polynomialOrder, dev_output);
    } else if (style == nondivtree) {
        cudaSum_nondivtree_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                dev_c, polynomialOrder, dev_output);
    } else if (style == constmem) {
        
        //initialize the constant memory
        cudaMemcpyToSymbol("constant_c", c, polynomialOrder * sizeof(float),
                0, cudaMemcpyHostToDevice);
        
        cudaSum_constmem_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                polynomialOrder, dev_output);
    } else {
        printf("Unknown style\n");
    }

    // Copy output from GPU
    cudaMemcpy(output, dev_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_input);
    cudaFree(dev_c);
    cudaFree(dev_output);
}
