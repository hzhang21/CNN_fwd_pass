#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *x, float *y, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int b = blockIdx.z;

    // ------------- MATRIX MULTIPLY -------------

    // matrix A dimensions (k4d)
    //int numARows = M;
    int numAColumns = C*K*K; // W_unroll

    // matrix B dimensions (X_unroll)
    //int numBRows = numAColumns;
    int numBColumns = H_out * W_out; // H_unroll

    // matrix C dimensions (y4d)
    int numCRows = M; //numARows;
    int numCColumns = numBColumns;

    int width = numAColumns; //numAColumns == numBRows == W_unroll == C * K * K

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];   //to hold filter-bank matrix,   k4d
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];   //to hold input features,       x4d

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float dot_product = 0;
    int x_c, x_h, x_w, x_p, x_q, x_pq;


    for(int tile = 0; tile < (width + TILE_WIDTH-1)/TILE_WIDTH; tile++) { //for tile in tiles
        if(tile*TILE_WIDTH + threadIdx.x < width && row < numCRows) {
            tileA[threadIdx.y][threadIdx.x] = k[row*numAColumns + tile*TILE_WIDTH+threadIdx.x]; //TODO  (m,c,p,q)
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if(tile*TILE_WIDTH + threadIdx.y < width && col < numCColumns) {
            x_c = (tile*TILE_WIDTH+threadIdx.y)/(K*K);
            x_h = col/W_out;
            x_w = col%W_out; // = w_idx
            x_pq = (tile*TILE_WIDTH+threadIdx.y)%(K*K); // = p*K + q
            x_p = x_pq/K;
            x_q = x_pq%K;
            tileB[threadIdx.y][threadIdx.x] = x4d(b, x_c, x_h + x_p, x_w + x_q); //X_unroll[(tile*TILE_WIDTH+threadIdx.y)*numBColumns+col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
    __syncthreads();

    for(int i = 0; i < TILE_WIDTH; i++) { //for element in tile
        dot_product += (tileA[threadIdx.y][i] * tileB[i][threadIdx.x]);
    }
    __syncthreads();

    }
    if(row < numCRows && col < numCColumns) {
        //C[row*numCColumns + col] = dot_product; //REPLACE WITH BELOW
        y[(b * M * H_out * W_out) + row*numCColumns + col] = dot_product;
    }

    // ------------- MATRIX MULTIPLY END -------------

#undef x4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(1.0*W_out/TILE_WIDTH);
    int H_grid = ceil(1.0*H_out/TILE_WIDTH);

    // ------------ADDITIONAL UNROLL CODE START -------------
    //int W_unroll = C * K * K;
    int H_unroll = H_out * W_out;

    // Set the kernel dimensions
    dim3 dimGrid(ceil(1.0*H_unroll/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH), B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    forward_kernel<<<dimGrid, dimBlock>>>(x.dptr_, y.dptr_,w.dptr_, B,M,C,H,W,K, W_grid);


    // ----------- ADDITIONAL CODE END ----------

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif

//grid dim (ceil(H_out*W_out/TILE_WIDTH), ceil(M/TILE_WIDTH), B)
