// P0b: raw DUMMA tiled GEMM ceiling. C[M,N] = A[M,K] @ B^T, B stored [N,K].
// fp16 inputs, fp32 mma accumulate (m16n16k16), fp32 C out.
// Tile: BM=128 BN=128 BK=32, 8 warps (2x4 -> warp tile 64x32), smem 2-stage.
// Correctness: small shape vs CPU. Then big-shape hipEvent timing (argv M N K).
#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK_HIP(call)                                                    \
  {                                                                        \
    hipError_t err = (call);                                               \
    if (err != hipSuccess) {                                               \
      std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line "  \
                << __LINE__ << std::endl;                                  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

#define BM 128
#define BN 128
#define BK 32
#define SLD (BK + 8)  // smem row stride, +8 halfs pad against bank conflicts

using namespace du;

template <bool NOLOAD>
__global__ __launch_bounds__(256) void dumma_gemm(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[2][BM][SLD];
  __shared__ __half Bs[2][BN][SLD];

  const int tid = threadIdx.x;
  const int warp = tid >> 6;  // WAVE64: 256 thr = 4 warps, DUMMA needs 64 lanes
  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  // warp tile 64x64 in a 2x2 warp grid (4 wave64 warps fill the 128x128 CTA)
  const int warp_m0 = (warp >> 1) * 64;
  const int warp_n0 = (warp & 1) * 64;

  // 16B-vectorized tile load: 8 halfs per thread; BM*BK = 512 chunks / 256 thr
  auto load_tile = [&](int buf, int k0) {
#pragma unroll
    for (int it = 0; it < 2; it++) {
      int c = tid + it * 256;
      int r = c >> 2, cc = (c & 3) << 3;
      *reinterpret_cast<uint4*>(&As[buf][r][cc]) =
          *reinterpret_cast<const uint4*>(&A[(long)(block_m + r) * K + k0 + cc]);
    }
#pragma unroll
    for (int it = 0; it < 2; it++) {
      int c = tid + it * 256;
      int r = c >> 2, cc = (c & 3) << 3;
      *reinterpret_cast<uint4*>(&Bs[buf][r][cc]) =
          *reinterpret_cast<const uint4*>(&B[(long)(block_n + r) * K + k0 + cc]);
    }
  };

  // accumulators: 4 (m) x 4 (n) 16x16 fp32 fragments per warp (64x64)
  dumma::DUFragment<dumma::accumulator, 16, 16, 16, float> d[4][4];
#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int nj = 0; nj < 4; nj++)
      dumma::du_fill_fragment(d[mi][nj], 0.0f);

  const int nk = K / BK;
  load_tile(0, 0);
  __syncthreads();
  for (int kk = 0; kk < nk; kk++) {
    const int cur = kk & 1, nxt = cur ^ 1;
    if (!NOLOAD && kk + 1 < nk) load_tile(nxt, (kk + 1) * BK);
    // compute stage cur: 2 k16 steps; a-frags row_major (k contiguous),
    // b-frags col_major over our [n][k] tile == math B^T
    dumma::DUFragment<dumma::matrix_a, 16, 16, 16, __half,
                      dumma::row_major>
        af[4];
    dumma::DUFragment<dumma::matrix_b, 16, 16, 16, __half,
                      dumma::col_major>
        bf[4];
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks++) {
#pragma unroll
      for (int mi = 0; mi < 4; mi++)
        dumma::du_load_matrix_sync(
            af[mi], &As[cur][warp_m0 + mi * 16][ks * 16], SLD);
#pragma unroll
      for (int nj = 0; nj < 4; nj++)
        dumma::du_load_matrix_sync(
            bf[nj], &Bs[cur][warp_n0 + nj * 16][ks * 16], SLD);
#pragma unroll
      for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int nj = 0; nj < 4; nj++)
          dumma::du_mma_sync(d[mi][nj], af[mi], bf[nj], d[mi][nj]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int nj = 0; nj < 4; nj++)
      dumma::du_store_matrix_sync(
          &C[(long)(block_m + warp_m0 + mi * 16) * N + block_n + warp_n0 +
             nj * 16],
          d[mi][nj], N, dumma::mem_row_major);
}

int main(int argc, char** argv) {
  bool NOLOAD = getenv("NOLOAD") != nullptr;
  int M = argc > 1 ? atoi(argv[1]) : 8192;
  int N = argc > 2 ? atoi(argv[2]) : 29568;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  std::cout << "P0b shape M=" << M << " N=" << N << " K=" << K << std::endl;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // ---------- correctness on a small shape (same code path, nk=8) ----------
  {
    int m = 256, n = 384, k = 256;  // 2x3 CTAs
    std::vector<__half> hA(m * k), hB(n * k);
    std::vector<float> hC(m * n, 0.f), hD(m * n);
    for (auto& x : hA) x = __float2half(dis(gen));
    for (auto& x : hB) x = __float2half(dis(gen));
    __half *dA, *dB;
    float* dC;
    CHECK_HIP(hipMalloc(&dA, hA.size() * 2));
    CHECK_HIP(hipMalloc(&dB, hB.size() * 2));
    CHECK_HIP(hipMalloc(&dC, hC.size() * 4));
    CHECK_HIP(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, hC.data(), hC.size() * 4, hipMemcpyHostToDevice));
    dim3 grid(n / BN, m / BM);
    dumma_gemm<false><<<grid, 256>>>(dA, dB, dC, m, n, k);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipMemcpy(hD.data(), dC, hD.size() * 4, hipMemcpyDeviceToHost));
    double max_abs = 0.0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        float sum = 0.f;
        for (int kk = 0; kk < k; kk++)
          sum += __half2float(hA[i * k + kk]) * __half2float(hB[j * k + kk]);
        max_abs = std::fmax(max_abs, fabs((double)hD[i * n + j] - sum));
      }
    std::cout << (max_abs < 1e-2 ? "P0B CORRECT" : "P0B WRONG") << " max_abs="
              << max_abs << std::endl;
    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC));
  }

  // ---------- big-shape timing ----------
  std::vector<__half> hA(M * K), hB(N * K);
  for (auto& x : hA) x = __float2half(dis(gen));
  for (auto& x : hB) x = __float2half(dis(gen));
  __half *dA, *dB;
  float* dC;
  CHECK_HIP(hipMalloc(&dA, (size_t)M * K * 2));
  CHECK_HIP(hipMalloc(&dB, (size_t)N * K * 2));
  CHECK_HIP(hipMalloc(&dC, (size_t)M * N * 4));
  CHECK_HIP(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(dC, 0, (size_t)M * N * 4));

  dim3 grid(N / BN, M / BM);
  for (int w = 0; w < 3; w++)
    if (NOLOAD) dumma_gemm<true><<<grid, 256>>>(dA, dB, dC, M, N, K);
    else dumma_gemm<false><<<grid, 256>>>(dA, dB, dC, M, N, K);
  CHECK_HIP(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  CHECK_HIP(hipEventCreate(&e0));
  CHECK_HIP(hipEventCreate(&e1));
  CHECK_HIP(hipEventRecord(e0));
  const int ITERS = 10;
  for (int it = 0; it < ITERS; it++)
    if (NOLOAD) dumma_gemm<true><<<grid, 256>>>(dA, dB, dC, M, N, K);
    else dumma_gemm<false><<<grid, 256>>>(dA, dB, dC, M, N, K);
  CHECK_HIP(hipEventRecord(e1));
  CHECK_HIP(hipEventSynchronize(e1));
  CHECK_HIP(hipGetLastError());
  float ms = 0.f;
  CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
  ms /= ITERS;
  double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
  printf("P0B TIME %.3f ms/iter  TFLOPS %.1f  (grid %dx%d)\n", ms, tflops,
         grid.x, grid.y);
  return 0;
}
