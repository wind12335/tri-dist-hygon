// P0a: DTK26.04 DUMMA manual example (fp32 m16n16k8) — toolchain check.
#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <iostream>
#include <random>
#include <iomanip>

#define M 16
#define N 16
#define K 8

using namespace du;

#define CHECK_HIP(call)                                                    \
  {                                                                        \
    hipError_t err = (call);                                               \
    if (err != hipSuccess) {                                               \
      std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line "  \
                << __LINE__ << std::endl;                                  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

__global__ void tensorCoreMatmulAddCKernel(const float* A, const float* B,
                                           const float* C, float* D) {
  dumma::DUFragment<dumma::matrix_a, M, N, K, float, dumma::row_major> a_frag;
  dumma::DUFragment<dumma::matrix_b, M, N, K, float, dumma::row_major> b_frag;
  dumma::DUFragment<dumma::accumulator, M, N, K, float> c_frag;
  dumma::DUFragment<dumma::accumulator, M, N, K, float> d_frag;

  dumma::du_load_matrix_sync(a_frag, A, K);
  dumma::du_load_matrix_sync(b_frag, B, N);
  dumma::du_load_matrix_sync(c_frag, C, N, dumma::mem_row_major);
  dumma::du_mma_sync(d_frag, a_frag, b_frag, c_frag);
  dumma::du_store_matrix_sync(D, d_frag, M, dumma::mem_row_major);
}

int main() {
  float *h_A = new float[M * K], *h_B = new float[K * N];
  float *h_C = new float[M * N], *h_D = new float[M * N];
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < M * K; i++) h_A[i] = dis(gen);
  for (int i = 0; i < K * N; i++) h_B[i] = dis(gen);
  for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

  float *d_A, *d_B, *d_C, *d_D;
  CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_D, M * N * sizeof(float)));
  CHECK_HIP(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_C, h_C, M * N * sizeof(float), hipMemcpyHostToDevice));

  tensorCoreMatmulAddCKernel<<<dim3(1), dim3(64)>>>(d_A, d_B, d_C, d_D);
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipMemcpy(h_D, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference D = A*B (row-major operands)
  double max_abs = 0.0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += h_A[i * K + k] * h_B[k * N + j];
      double d = fabs((double)h_D[i * N + j] - (double)sum);
      max_abs = fmax(max_abs, d);
    }
  std::cout << (max_abs < 1e-3 ? "P0A PASS" : "P0A FAIL") << " max_abs="
            << max_abs << std::endl;
  return 0;
}
