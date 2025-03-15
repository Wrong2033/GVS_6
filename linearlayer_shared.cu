#include <torch/extension.h>

template<typename T>
using accessor_2d = torch::PackedTensorAccessor32<T,2>;

template<typename T>
using accessor_1d = torch::PackedTensorAccessor32<T,1>;

const int block_size = 16;

__global__ void linear_function_shared(accessor_2d<float> x,
                                       accessor_2d<float> w,
                                       accessor_2d<float> y,
                                       accessor_1d<float> b) {
    __shared__ float x_shared[block_size][block_size];
    __shared__ float w_shared[block_size][block_size];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < x.size(1); k_tile += block_size) {
        int x_k = k_tile + threadIdx.x;
        if (row < x.size(0) && x_k < x.size(1)) {
            x_shared[threadIdx.y][threadIdx.x] = x[row][x_k];
        } else {
            x_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int w_k = k_tile + threadIdx.y;
        if (col < w.size(0) && w_k < w.size(1)) {
            w_shared[threadIdx.y][threadIdx.x] = w[col][w_k];
        } else {
            w_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
            acc += x_shared[threadIdx.y][i] * w_shared[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < x.size(0) && col < w.size(0)) {
        y[row][col] = acc + b[col];
    }
}

__global__ void arr_mult_2d_shared(accessor_2d<float> a,
                                   accessor_2d<float> b,
                                   accessor_2d<float> c) {
    __shared__ float a_shared[block_size][block_size];
    __shared__ float b_shared[block_size][block_size];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < a.size(1); k_tile += block_size) {
        int a_k = k_tile + threadIdx.x;
        if (row < a.size(0) && a_k < a.size(1)) {
            a_shared[threadIdx.y][threadIdx.x] = a[row][a_k];
        } else {
            a_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_k = k_tile + threadIdx.y;
        if (col < b.size(1) && b_k < b.size(0)) {
            b_shared[threadIdx.y][threadIdx.x] = b[b_k][col];
        } else {
            b_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
            acc += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < a.size(0) && col < b.size(1)) {
        c[row][col] = acc;
    }
}

__global__ void arr_mult_2d_trans_shared(accessor_2d<float> a,
                                         accessor_2d<float> b,
                                         accessor_2d<float> c) {
    __shared__ float a_shared[block_size][block_size];
    __shared__ float b_shared[block_size][block_size];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < a.size(0); k_tile += block_size) {
        int a_k = k_tile + threadIdx.x;
        if (a_k < a.size(0) && row < a.size(1)) {
            a_shared[threadIdx.y][threadIdx.x] = a[a_k][row];
        } else {
            a_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_k = k_tile + threadIdx.y;
        if (b_k < b.size(0) && col < b.size(1)) {
            b_shared[threadIdx.y][threadIdx.x] = b[b_k][col];
        } else {
            b_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
            acc += a_shared[i][threadIdx.y] * b_shared[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < c.size(0) && col < c.size(1)) {
        c[row][col] = acc;
    }
}

__global__ void sum_for_bias(accessor_2d<float> x,
                             accessor_1d<float> y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int n = x.size(0);
    int m = x.size(1);

    if (i < m) {
        float acc = 0;
        for (int j = 0; j < n; j++) {
            acc += x[j][i];
        }
        y[i] = acc;
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__forceinline__ int calc_grid_size(int m) {
    return (m + block_size - 1) / block_size;
}

torch::Tensor forward_linear(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);

    int n = b.numel();
    int k = w.numel() / n;
    int m = x.numel() / k;

    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA).requires_grad(true);
    torch::Tensor y = torch::zeros({m, n}, options);

    dim3 dimGrid(calc_grid_size(n), calc_grid_size(m));
    dim3 dimBlock(block_size, block_size);
    linear_function_shared<<<dimGrid, dimBlock>>>(
        x.packed_accessor32<float, 2>(),
        w.packed_accessor32<float, 2>(),
        y.packed_accessor32<float, 2>(),
        b.packed_accessor32<float, 1>()
    );

    return y;
}

std::vector<torch::Tensor> backward_linear(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);
    CHECK_INPUT(y);

    auto y_pa = y.packed_accessor32<float, 2>();
    auto x_pa = x.packed_accessor32<float, 2>();
    auto w_pa = w.packed_accessor32<float, 2>();
    auto b_pa = b.packed_accessor32<float, 1>();

    int m = x_pa.size(0);
    int n = y_pa.size(1);
    int k = x_pa.size(1);

    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA).requires_grad(true);
    torch::Tensor grad_input = torch::zeros({m, k}, options);
    torch::Tensor grad_weight = torch::zeros({n, k}, options);
    torch::Tensor grad_bias = torch::zeros({n}, options);

    dim3 dimGrid(calc_grid_size(k), calc_grid_size(m));
    dim3 dimBlock(block_size, block_size);
    arr_mult_2d_shared<<<dimGrid, dimBlock>>>(
        y_pa,
        w_pa,
        grad_input.packed_accessor32<float, 2>()
    );

    dim3 dimGrid2(calc_grid_size(k), calc_grid_size(n));
    arr_mult_2d_trans_shared<<<dimGrid2, dimBlock>>>(
        y_pa,
        x_pa,
        grad_weight.packed_accessor32<float, 2>()
    );

    sum_for_bias<<<calc_grid_size(n), block_size>>>(
        y_pa,
        grad_bias.packed_accessor32<float, 1>()
    );

    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_forward_linear", &forward_linear, "Custom forward with shared memory");
    m.def("my_backward_linear", &backward_linear, "Custom backward with shared memory");
}
