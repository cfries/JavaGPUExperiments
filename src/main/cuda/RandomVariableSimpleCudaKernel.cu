extern "C"
__global__ void cuAdd(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] + b[i];
    }

}

extern "C"
__global__ void cuMult(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] * b[i];
    }

}

extern "C"
__global__ void cuDiv(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] / b[i];
    }

}

extern "C"
__global__ void cuExp(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = expf(a[i]);
    }

}

